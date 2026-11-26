#!/usr/bin/env python3
"""pdfsearch - powerful search over encrypted (or plain) PDFs.

Not just Ctrl+F: results are ranked passages scored by a hybrid of

  * BM25 over word unigrams + bigrams          -> strong lexical/phrase ranking
  * TF-IDF cosine over character 3-5 grams     -> typo tolerance, hyphenation,
                                                  line breaks, OCR noise, partial words
  * fuzzy correction of query terms against the document vocabulary
  * simple morphological variants (plural/singular)

so misspelled, partial, or reworded queries still surface the right passages.

Dependencies: pypdf, numpy. Optional: rapidfuzz (faster fuzzy matching).

Usage:
  pdfsearch.py info FILE.pdf [--password PW]...
  pdfsearch.py index FILE.pdf [FILE.pdf...] --out INDEX [--password PW]...
  pdfsearch.py search FILE.pdf [FILE.pdf...] --query "..." [--password PW]...
  pdfsearch.py search --index INDEX --query "..." [--top-k 10] [--json]

Passwords are tried in order: "" (empty), then --password values, then the
PDFSEARCH_PASSWORDS environment variable (os.pathsep-separated). If the PDF is
encrypted and no password works, you are prompted when running interactively.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from pypdf import PdfReader

try:
    from rapidfuzz import fuzz, process

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    import difflib

    _HAS_RAPIDFUZZ = False

try:
    from pypdf import PasswordType

    def _decrypted(result) -> bool:
        return result != PasswordType.NOT_DECRYPTED
except ImportError:  # pragma: no cover

    def _decrypted(result) -> bool:
        return int(result) != 0


WORD_RE = re.compile(r"[a-z0-9]+")

CHUNK_MAX = 600
CHUNK_STEP = 450
CHAR_NGRAM_SIZES = (3, 4, 5)
W_BM25 = 0.62
W_CHAR = 0.38
PHRASE_BONUS = 0.25


class PasswordError(Exception):
    pass


def _enforce_offline():
    """Hard runtime guarantee: block all outbound network from this process.

    Patches TCP/UDP connect paths and DNS resolution so no code path (ours or
    any dependency's) can send PDF content anywhere. Disable only with
    --allow-network.
    """
    import socket

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "pdfsearch runs fully offline: outbound network access is disabled "
            "(use --allow-network to override)"
        )

    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    socket.gethostbyname = _blocked
    socket.gethostbyname_ex = _blocked


# --------------------------------------------------------------------------- #
# text normalization
# --------------------------------------------------------------------------- #
def fold_with_map(s: str):
    """Lowercase + unicode-normalize + strip accents, keeping a map from each
    folded char back to its source index so highlights land on the original."""
    chars: list[str] = []
    imap: list[int] = []
    for i, ch in enumerate(s):
        f = unicodedata.normalize("NFKC", ch).casefold()
        f = "".join(
            c for c in unicodedata.normalize("NFD", f)
            if unicodedata.category(c) != "Mn"
        )
        for c in f:
            chars.append(c)
            imap.append(i)
    return "".join(chars), imap


def word_tokens(folded: str) -> list[str]:
    return WORD_RE.findall(folded)


def word_features(tokens: list[str]) -> list[str]:
    feats = list(tokens)
    feats.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return feats


def char_ngrams(folded: str, sizes=CHAR_NGRAM_SIZES) -> Counter:
    grams: Counter = Counter()
    flat = re.sub(r"\s+", " ", folded)
    for n in sizes:
        if len(flat) >= n:
            for i in range(len(flat) - n + 1):
                grams[flat[i : i + n]] += 1
    return grams


def morph_variants(t: str) -> list[str]:
    out = []
    if t.endswith("ies") and len(t) > 4:
        out.append(t[:-3] + "y")
    if t.endswith("es") and len(t) > 3:
        out.append(t[:-2])
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        out.append(t[:-1])
    return out


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #
def collect_passwords(args) -> list[str]:
    pws: list[str] = [""]
    for p in getattr(args, "password", None) or []:
        pws.append(p)
    env = os.environ.get("PDFSEARCH_PASSWORDS")
    if env:
        pws.extend(p for p in env.split(os.pathsep) if p)
    seen: set[str] = set()
    out: list[str] = []
    for p in pws:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def open_pdf(path: str, passwords: list[str], allow_prompt: bool = True):
    """Return (reader, used_password or None). Raises PasswordError."""
    reader = PdfReader(path)
    if not reader.is_encrypted:
        return reader, None
    for pw in passwords:
        try:
            if _decrypted(reader.decrypt(pw)):
                return reader, (pw or None)
        except Exception:
            continue
    if allow_prompt and sys.stdin.isatty():
        try:
            pw = getpass.getpass(f"Password for {path}: ")
        except (EOFError, KeyboardInterrupt):
            pw = ""
        if pw:
            try:
                if _decrypted(reader.decrypt(pw)):
                    return reader, pw
            except Exception:
                pass
    raise PasswordError(f"{path}: encrypted and none of the supplied passwords worked")


def extract_pages(path: str, passwords: list[str]):
    reader, used_pw = open_pdf(path, passwords)
    pages: list[str] = []
    for page in reader.pages:
        txt = ""
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if len(txt.strip()) < 20:
            try:
                txt = page.extract_text(extraction_mode="layout") or txt
            except Exception:
                pass
        pages.append(txt)
    meta = {}
    try:
        m = reader.metadata
        if m:
            meta = {k: str(v) for k, v in dict(m).items() if v is not None}
    except Exception:
        pass
    return pages, used_pw, meta, len(reader.pages)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, max_len=CHUNK_MAX, step=CHUNK_STEP):
    """Overlapping windows snapped to word boundaries. Returns (start, chunk)."""
    chunks: list[tuple[int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + max_len, n)
        if end < n:
            sp = text.rfind(" ", i + max_len // 2, end)
            if sp > i:
                end = sp
        piece = text[i:end].strip()
        if piece:
            chunks.append((i, piece))
        if end >= n:
            break
        nxt = end - (max_len - step)
        sp = text.find(" ", nxt)
        if sp != -1 and sp < end:
            nxt = sp + 1
        if nxt <= i:
            nxt = end
        i = nxt
    return chunks


# --------------------------------------------------------------------------- #
# indexes
# --------------------------------------------------------------------------- #
class BM25Index:
    def __init__(self, docs_feats: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_feats)
        self.dl = np.array([len(d) for d in docs_feats], dtype=float)
        self.avgdl = float(self.dl.mean()) or 1.0
        df: Counter = Counter()
        for d in docs_feats:
            df.update(set(d))
        self.df = df
        self.vocab = set(df)
        self.idf = {
            t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        post: dict[str, tuple[list[int], list[float]]] = {}
        for i, d in enumerate(docs_feats):
            for t, f in Counter(d).items():
                bucket = post.setdefault(t, ([], []))
                bucket[0].append(i)
                bucket[1].append(float(f))
        self.post = {
            t: (np.asarray(ids, dtype=np.int64), np.asarray(fs, dtype=float))
            for t, (ids, fs) in post.items()
        }

    def score(self, qterms: dict[str, float]) -> np.ndarray:
        s = np.zeros(self.n, dtype=float)
        for t, qw in qterms.items():
            p = self.post.get(t)
            if p is None:
                continue
            ids, tf = p
            denom = tf + self.k1 * (1.0 - self.b + self.b * self.dl[ids] / self.avgdl)
            s[ids] += qw * self.idf[t] * (tf * (self.k1 + 1.0)) / denom
        return s


class CharIndex:
    def __init__(self, docs_folded: list[str], min_df=2, max_features=250_000):
        self.sizes = CHAR_NGRAM_SIZES
        self.n = len(docs_folded)
        per_doc = [char_ngrams(s) for s in docs_folded]
        df: Counter = Counter()
        for g in per_doc:
            df.update(g.keys())
        keep = [t for t, c in df.items() if c >= min_df]
        keep.sort(key=lambda t: -df[t])
        del keep[max_features:]
        self.vocab = set(keep)
        self.idf = {
            t: math.log(1.0 + (self.n - df[t] + 0.5) / (df[t] + 0.5)) for t in keep
        }
        post: dict[str, tuple[list[int], list[float]]] = {}
        for i, grams in enumerate(per_doc):
            w: dict[str, float] = {}
            for t, f in grams.items():
                if t in self.vocab:
                    w[t] = (1.0 + math.log(f)) * self.idf[t]
            norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
            for t, v in w.items():
                bucket = post.setdefault(t, ([], []))
                bucket[0].append(i)
                bucket[1].append(v / norm)
        self.post = {
            t: (np.asarray(ids, dtype=np.int64), np.asarray(ws, dtype=float))
            for t, (ids, ws) in post.items()
        }

    def score(self, q_folded: str) -> np.ndarray:
        qg = char_ngrams(q_folded, self.sizes)
        qw: dict[str, float] = {}
        for t, f in qg.items():
            if t in self.idf:
                qw[t] = (1.0 + math.log(f)) * self.idf[t]
        qn = math.sqrt(sum(v * v for v in qw.values())) or 1.0
        s = np.zeros(self.n, dtype=float)
        for t, v in qw.items():
            p = self.post.get(t)
            if p is None:
                continue
            ids, ws = p
            s[ids] += (v / qn) * ws
        return s


# --------------------------------------------------------------------------- #
# search index over one or more PDFs
# --------------------------------------------------------------------------- #
class SearchIndex:
    def __init__(self):
        self.passages: list[dict] = []
        self.bm25: BM25Index | None = None
        self.char: CharIndex | None = None
        self.vocab_list: list[str] = []

    # -- building ---------------------------------------------------------- #
    @classmethod
    def build(cls, paths: list[str], passwords: list[str]) -> "SearchIndex":
        idx = cls()
        for path in paths:
            pages, _used, _meta, _npages = extract_pages(path, passwords)
            for pno, text in enumerate(pages, start=1):
                for start, chunk in chunk_text(text):
                    fold, imap = fold_with_map(chunk)
                    toks = word_tokens(fold)
                    if not toks and len(fold.strip()) < 3:
                        continue
                    idx.passages.append(
                        {
                            "doc": path,
                            "page": pno,
                            "start": start,
                            "text": chunk,
                            "fold": fold,
                            "imap": np.asarray(imap, dtype=np.int32),
                            "tokens": toks,
                            "feats": word_features(toks),
                        }
                    )
        if not idx.passages:
            raise SystemExit("no extractable text found (scanned/image-only PDF?)")
        idx.bm25 = BM25Index([p["feats"] for p in idx.passages])
        idx.char = CharIndex([p["fold"] for p in idx.passages])
        idx.vocab_list = sorted(idx.bm25.vocab)
        return idx

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SearchIndex":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise SystemExit(f"{path}: not a pdfsearch index")
        return obj

    # -- query expansion --------------------------------------------------- #
    def _best_match(self, term: str) -> str | None:
        if _HAS_RAPIDFUZZ:
            res = process.extractOne(
                term, self.vocab_list, scorer=fuzz.ratio, score_cutoff=82
            )
            if res is None:
                return None
            best, score, _ = res
            return best if score >= 82 else None
        matches = difflib.get_close_matches(term, self.vocab_list, n=1, cutoff=0.82)
        return matches[0] if matches else None

    def _expand(self, q_tokens: list[str]) -> dict[str, float]:
        qw: dict[str, float] = {}
        for t in q_tokens:
            qw[t] = qw.get(t, 0.0) + 2.0
        for t in q_tokens:
            if len(t) >= 4 and t not in self.bm25.vocab:
                best = self._best_match(t)
                if best:
                    qw[best] = qw.get(best, 0.0) + 1.0
            for v in morph_variants(t):
                if v in self.bm25.vocab and v != t:
                    qw[v] = qw.get(v, 0.0) + 1.0
        for a, b in zip(q_tokens, q_tokens[1:]):
            qw[f"{a} {b}"] = qw.get(f"{a} {b}", 0.0) + 1.5
        return qw

    # -- search ------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.05,
        pages: set[int] | None = None,
    ) -> list[dict]:
        q_fold, _ = fold_with_map(query)
        q_tokens = word_tokens(q_fold)
        if not q_tokens:
            return []
        qw = self._expand(q_tokens)

        bm = self.bm25.score(qw)
        ch = self.char.score(q_fold)
        bm_max = bm.max()
        ch_max = ch.max()
        scores = np.zeros(len(self.passages))
        if bm_max > 0:
            scores += W_BM25 * bm / bm_max
        if ch_max > 0:
            scores += W_CHAR * ch / ch_max

        phrase = " ".join(q_tokens)
        if len(q_tokens) > 1:
            for i, p in enumerate(self.passages):
                if phrase in p["fold"]:
                    scores[i] += PHRASE_BONUS

        order = np.argsort(-scores)
        results: list[dict] = []
        for i in order:
            if len(results) >= top_k:
                break
            s = float(scores[i])
            if s < min_score:
                break
            p = self.passages[i]
            if pages is not None and p["page"] not in pages:
                continue
            hi, plain = self._snippet(p, q_tokens, qw)
            results.append(
                {
                    "doc": p["doc"],
                    "page": p["page"],
                    "score": round(s, 4),
                    "snippet": hi,
                    "snippet_plain": plain,
                    "start": p["start"],
                }
            )
        return results

    def _snippet(self, p: dict, q_tokens: list[str], qw: dict[str, float]):
        fold: str = p["fold"]
        imap: np.ndarray = p["imap"]
        orig: str = p["text"]
        terms = [t for t in qw if len(t) >= 2 and " " not in t]
        spans: list[tuple[int, int]] = []
        for t in terms:
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
            for m in pat.finditer(fold):
                spans.append((m.start(), m.end()))
        if len(q_tokens) > 1:
            ph = re.sub(r"\s+", " ", fold_with_map(" ".join(q_tokens))[0])
            m = fold.find(ph)
            if m != -1:
                spans.append((m, m + len(ph)))
        context = 180
        if spans:
            centers = sorted((a + b) / 2 for a, b in spans)
            c = centers[len(centers) // 2]
            lo = max(0, int(c - context // 2))
            hi = min(len(fold), int(c + context // 2))
        else:
            lo, hi = 0, min(len(fold), context)
        oa = int(imap[lo]) if lo < len(imap) else 0
        ob = int(imap[hi - 1]) + 1 if hi > lo and hi - 1 < len(imap) else len(orig)
        ob = min(ob, len(orig))
        while oa > 0 and orig[oa - 1] not in " \n\t":
            oa -= 1
        while ob < len(orig) and orig[ob] not in " \n\t":
            ob += 1
        mask = bytearray(len(orig))
        for a, b in spans:
            pa = int(imap[a]) if a < len(imap) else len(orig)
            pb = int(imap[b - 1]) + 1 if b - 1 < len(imap) else len(orig)
            for j in range(max(0, pa), min(pb, len(orig))):
                mask[j] = 1
        out: list[str] = []
        if oa > 0:
            out.append("...")
        in_hl = False
        for j in range(oa, ob):
            if mask[j] and not in_hl:
                out.append("**")
                in_hl = True
            elif not mask[j] and in_hl:
                out.append("**")
                in_hl = False
            out.append(orig[j])
        if in_hl:
            out.append("**")
        if ob < len(orig):
            out.append("...")
        return "".join(out), orig[oa:ob].strip()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_password_arg(p: argparse.ArgumentParser):
    p.add_argument(
        "--password",
        action="append",
        default=[],
        metavar="PW",
        help="PDF password (repeatable). Also reads PDFSEARCH_PASSWORDS env var.",
    )


def parse_pages(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "dim": "\033[2m",
}


def _paint(text: str, *styles: str, enabled: bool) -> str:
    if not enabled or not styles:
        return text
    return "".join(ANSI[s] for s in styles) + text + ANSI["reset"]


def cmd_info(args) -> int:
    passwords = collect_passwords(args)
    for path in args.pdfs:
        reader = PdfReader(path)
        encrypted = bool(reader.is_encrypted)
        info = {
            "file": path,
            "encrypted": encrypted,
            "pages": None,
            "password_works": None,
            "metadata": {},
        }
        try:
            _pages, used, meta, npages = extract_pages(path, passwords)
            info["pages"] = npages
            info["password_works"] = (
                used
                if used is not None
                else ("<empty password>" if encrypted else "<not encrypted>")
            )
            info["metadata"] = meta
        except PasswordError as e:
            info["password_works"] = False
            print(f"{path}: {e}", file=sys.stderr)
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"file:     {info['file']}")
            print(f"encrypted: {info['encrypted']}")
            print(f"pages:     {info['pages']}")
            print(f"access:    {info['password_works']}")
            for k, v in info["metadata"].items():
                print(f"meta {k}: {v}")
    return 0


def cmd_index(args) -> int:
    passwords = collect_passwords(args)
    idx = SearchIndex.build(args.pdfs, passwords)
    idx.save(args.out)
    print(f"indexed {len(idx.passages)} passages from {len(args.pdfs)} file(s) -> {args.out}")
    return 0


def cmd_search(args) -> int:
    color = sys.stdout.isatty() and not args.no_color
    if args.index:
        idx = SearchIndex.load(args.index)
    else:
        passwords = collect_passwords(args)
        idx = SearchIndex.build(args.pdfs, passwords)
    results = idx.search(
        args.query, top_k=args.top_k, min_score=args.min_score, pages=parse_pages(args.pages)
    )
    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "count": len(results),
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if not results:
        print(f'no matches for "{args.query}"')
        return 1
    print(f'{len(results)} match(es) for "{args.query}"\n')
    for rank, r in enumerate(results, start=1):
        head = _paint(
            f"#{rank}  page {r['page']}  score {r['score']:.3f}  {Path(r['doc']).name}",
            "bold",
            enabled=color,
        )
        print(head)
        if color:
            parts = r["snippet"].replace("**", "\x00").split("\x00")
            line = parts[0]
            for k, part in enumerate(parts[1:], start=1):
                line += _paint(part, "bold", "yellow", enabled=True) if k % 2 == 1 else part
        else:
            line = r["snippet"]
        print("   " + line.replace("\n", " "))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pdfsearch",
        description="Powerful ranked search over encrypted or plain PDFs. "
        "Fully offline: outbound network is blocked at runtime by default.",
    )
    ap.add_argument(
        "--allow-network",
        action="store_true",
        help="disable the built-in offline guard (not recommended)",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show encryption status / metadata of PDFs")
    p_info.add_argument("pdfs", nargs="+")
    p_info.add_argument("--json", action="store_true")
    _add_password_arg(p_info)
    p_info.set_defaults(func=cmd_info)

    p_idx = sub.add_parser("index", help="build and save a search index")
    p_idx.add_argument("pdfs", nargs="+")
    p_idx.add_argument("--out", required=True, metavar="INDEX")
    _add_password_arg(p_idx)
    p_idx.set_defaults(func=cmd_index)

    p_s = sub.add_parser("search", help="search PDFs (or a saved index)")
    p_s.add_argument("pdfs", nargs="*", help="PDF file(s)")
    p_s.add_argument("--index", metavar="INDEX", help="use a saved index instead of PDFs")
    p_s.add_argument("-q", "--query", required=True)
    p_s.add_argument("--top-k", type=int, default=10)
    p_s.add_argument("--min-score", type=float, default=0.05)
    p_s.add_argument("--pages", metavar="e.g. 1-3,7", default=None)
    p_s.add_argument("--json", action="store_true")
    p_s.add_argument("--no-color", action="store_true")
    _add_password_arg(p_s)
    p_s.set_defaults(func=cmd_search)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_network:
        _enforce_offline()
    if args.command == "search" and not args.index and not args.pdfs:
        print("error: provide PDF file(s) or --index", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except PasswordError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
