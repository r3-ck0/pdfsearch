#!/usr/bin/env python3
"""End-to-end tests for pdfsearch.py (run: python3 tests/test_pdfsearch.py)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "pdfsearch.py"
FIX = ROOT / "tests" / "fixtures"
AES = FIX / "sample_aes256.pdf"
RC4 = FIX / "sample_rc4.pdf"
OWNER_ONLY = FIX / "sample_owner_only.pdf"
PLAIN = FIX / "sample.pdf"
PW = "s3cret-pass"


def _ensure_fixtures():
    if not AES.exists():
        try:
            import reportlab  # noqa: F401
        except ImportError:
            raise SystemExit("test fixtures need reportlab: pip install -r requirements-dev.txt")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_fixture

        make_fixture.main()


_ensure_fixtures()


def run(*args, env_extra=None, stdin_devnull=True):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL if stdin_devnull else None,
    )


def search_json(pdf, query, *extra):
    r = run("search", str(pdf), "-q", query, "--json", "--password", PW, *extra)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class TestPasswords(unittest.TestCase):
    def test_wrong_password_fails_cleanly(self):
        r = run("search", str(AES), "-q", "encryption", "--password", "nope")
        self.assertEqual(r.returncode, 2)
        self.assertIn("password", r.stderr.lower())

    def test_no_password_fails_cleanly(self):
        r = run("search", str(AES), "-q", "encryption")
        self.assertEqual(r.returncode, 2)

    def test_env_var_password(self):
        out = search_json_env(AES, "encryption")
        self.assertGreaterEqual(out["count"], 1)

    def test_owner_only_needs_no_password(self):
        r = run("search", str(OWNER_ONLY), "-q", "sourdough starter", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreaterEqual(json.loads(r.stdout)["count"], 1)

    def test_rc4_encrypted(self):
        out = search_json(RC4, "AES-256 symmetric encryption")
        self.assertEqual(out["results"][0]["page"], 1)

    def test_plain_pdf(self):
        r = run("search", str(PLAIN), "-q", "tomato sauce", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["results"][0]["page"], 6)

    def test_info_encrypted(self):
        r = run("info", str(AES), "--password", PW, "--json")
        data = json.loads(r.stdout)
        self.assertTrue(data["encrypted"])
        self.assertEqual(data["pages"], 6)


def search_json_env(pdf, query):
    r = run(
        "search", str(pdf), "-q", query, "--json",
        env_extra={"PDFSEARCH_PASSWORDS": PW},
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class TestSearchQuality(unittest.TestCase):
    def test_exact_phrase_top_result(self):
        out = search_json(AES, "semantic search understands the meaning")
        top = out["results"][0]
        self.assertEqual(top["page"], 1)
        self.assertIn("semantic search", top["snippet_plain"].lower())

    def test_typo_query_still_finds(self):
        out = search_json(AES, "semntic serch")
        top = out["results"][0]
        self.assertEqual(top["page"], 1)
        self.assertIn("semantic", top["snippet_plain"].lower())

    def test_partial_term(self):
        out = search_json(AES, "encrypt")
        pages = {r["page"] for r in out["results"][:3]}
        self.assertIn(1, pages)

    def test_phrase_ranking(self):
        out = search_json(AES, "gradient descent")
        self.assertEqual(out["results"][0]["page"], 2)  # page 2 mentions it twice

    def test_disambiguation(self):
        out = search_json(AES, "river bank erosion")
        self.assertEqual(out["results"][0]["page"], 4)  # not the financial bank page

    def test_reworded_query(self):
        out = search_json(AES, "money institution interest rates")
        self.assertEqual(out["results"][0]["page"], 5)

    def test_results_sorted_by_score(self):
        out = search_json(AES, "encryption password", "--top-k", "8")
        scores = [r["score"] for r in out["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_pages_filter(self):
        out = search_json(AES, "gradient descent", "--pages", "3")
        self.assertTrue(all(r["page"] == 3 for r in out["results"]))
        self.assertGreaterEqual(out["count"], 1)

    def test_snippet_highlight_markers(self):
        out = search_json(AES, "sourdough starter")
        self.assertIn("**", out["results"][0]["snippet"])

    def test_no_match(self):
        r = run("search", str(AES), "-q", "zzzzqqqq", "--password", PW)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no matches", r.stdout)


class TestIndexPersistence(unittest.TestCase):
    def test_index_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            idx = Path(td) / "x.pdix"
            r = run("index", str(AES), "--out", str(idx), "--password", PW)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(idx.exists())
            r = run("search", "--index", str(idx), "-q", "river bank erosion", "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertEqual(out["results"][0]["page"], 4)

    def test_multi_pdf_search(self):
        out = search_json_multi(
            [AES, OWNER_ONLY], "sourdough"
        )
        docs = {r["doc"] for r in out["results"]}
        self.assertEqual(docs, {str(AES), str(OWNER_ONLY)})


def search_json_multi(pdfs, query):
    args = ["search", *[str(p) for p in pdfs], "-q", query, "--json", "--password", PW]
    r = run(*args)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class TestOfflineGuarantee(unittest.TestCase):
    def test_network_blocked_in_process(self):
        script = (
            "import sys; sys.path.insert(0, %r);"
            "import socket, pdfsearch;"
            "pdfsearch._enforce_offline();"
            "checks = [lambda: socket.create_connection(('127.0.0.1', 80)),"
            "          lambda: socket.socket().connect(('127.0.0.1', 80)),"
            "          lambda: socket.getaddrinfo('docs.invalid', 80),"
            "          lambda: socket.gethostbyname('docs.invalid')];"
            "bad = 0\n"
            "for fn in checks:\n"
            "    try: fn()\n"
            "    except RuntimeError: pass\n"
            "    else: bad += 1\n"
            "sys.exit(1 if bad else 0)"
            % str(ROOT)
        )
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_search_works_with_offline_guard_default(self):
        # main() enforces the guard unless --allow-network is passed; a normal
        # search must still succeed with the guard active.
        r = run("search", str(AES), "-q", "semantic search", "--password", PW, "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreaterEqual(json.loads(r.stdout)["count"], 1)

    def test_tool_has_no_network_code(self):
        src = TOOL.read_text()
        for banned in ("urllib", "requests", "urlopen", "http://", "https://"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
