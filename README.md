# pdfsearch — powerful search over encrypted PDFs

A single-file Python CLI that decrypts password-protected PDFs (RC4 and AES,
including AES-256) and searches them with ranked, fuzzy-tolerant retrieval —
much stronger than Ctrl+F, fully offline.

## Why it's stronger than Ctrl+F

Every page is split into overlapping passages, and each query is scored by a
hybrid of:

| Signal | What it catches |
|---|---|
| BM25 over word unigrams + bigrams | ranked relevance, phrases, partial matches |
| TF-IDF cosine over character 3–5 grams | typos, hyphenation/line breaks, OCR noise, word fragments |
| Fuzzy query correction vs. document vocabulary (rapidfuzz) | misspelled query words (`semntic` → `semantic`) |
| Morphological variants | `databases` ↔ `database`, `policies` ↔ `policy` |
| Exact-phrase bonus | quoted-phrase-style queries rank highest |

Results are ranked passages with page numbers and highlighted snippets.

## Install (fully offline)

The tool makes **zero network calls**. On top of having no network code, it
blocks sockets and DNS resolution inside its own process at runtime, so no
code path can send PDF content anywhere. Pass `--allow-network` only if you
insist on disabling that guard.

To install on an air-gapped machine, build a local wheelhouse once on a
connected machine, then install with `--no-index`:

```bash
# on a connected machine, once:
pip download -r requirements.txt -d wheels/
# transfer wheels/ to the offline machine, then:
pip install --no-index --find-links=wheels -r requirements.txt
```

Runtime requirements: `pypdf`, `numpy` (see `requirements.txt`);
`rapidfuzz` optional (faster fuzzy matching); `cryptography` needed by pypdf
for AES-encrypted PDFs.

### Handling of sensitive data

- The decrypted text never leaves the process: no telemetry, no cloud calls,
  no temp uploads.
- A saved index (`.pdix`) **contains the decrypted plaintext** of the PDF. Store
  it with the same protection as the unencrypted document and delete it when
  done. Only load `.pdix` files you trust (they are pickles).
- `--password` on the command line is visible to other local users via `ps`
  and shell history. Prefer the interactive prompt or `PDFSEARCH_PASSWORDS`.

## Usage

```bash
# search an encrypted PDF (prompts for password if interactive)
python3 pdfsearch.py search report.pdf -q "gradient descent"

# with a password (repeatable), or via env var:
python3 pdfsearch.py search report.pdf -q "river bank erosion" --password s3cret
PDFSEARCH_PASSWORDS="pw1:pw2" python3 pdfsearch.py search report.pdf -q "AES"

# options
python3 pdfsearch.py search report.pdf -q "..." --top-k 20 --min-score 0.1 \
    --pages 1-5,7 --json --no-color

# build a reusable index once, search it many times (fast)
python3 pdfsearch.py index report.pdf --out report.pdix --password s3cret
python3 pdfsearch.py search --index report.pdix -q "semantic search"

# inspect a PDF: encryption status, page count, metadata
python3 pdfsearch.py info report.pdf [--password s3cret]

# search several PDFs at once
python3 pdfsearch.py search a.pdf b.pdf c.pdf -q "..." --password s3cret
```

Passwords are tried in order: empty password, then each `--password`, then
`PDFSEARCH_PASSWORDS` (os.pathsep-separated). Wrong/absent passwords exit with
code 2 and a clear error.

## Output

Human-readable by default (`**bold**` highlights, page + score per hit), or
`--json` for scripting:

```json
{
  "query": "semntic serch",
  "count": 3,
  "results": [
    {"doc": "report.pdf", "page": 1, "score": 1.0,
     "snippet": "...many **semantic** **search** systems...",
     "snippet_plain": "...many semantic search systems...", "start": 0}
  ]
}
```

## Notes / limits

- Text is extracted with pypdf; scanned image-only PDFs yield no text (OCR is
  out of scope). Encrypted PDFs need the `cryptography` package for AES.
- Snippets are shown with original casing; matching is case/accent-insensitive.
- Index files (`.pdix`) are pickles of the built index — only load indexes you
  trust, and never store passwords in them (they aren't).

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 tests/test_pdfsearch.py   # auto-generates encrypted sample PDFs on first run
```

`tests/make_fixture.py` can also be run standalone to (re)generate the fixtures
under `tests/fixtures/` (gitignored). The suite covers encrypted-PDF access
(AES-256/RC4/owner-only), wrong-password handling, fuzzy/reworded queries,
ranking, JSON output, index persistence, and the offline network guard.
