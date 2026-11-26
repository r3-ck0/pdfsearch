#!/usr/bin/env python3
"""Create sample encrypted PDFs for testing pdfsearch."""
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)

PAGES = [
    (
        "Introduction to Modern Encryption",
        [
            "Modern encryption protects documents at rest and in transit. AES-256 is the "
            "current standard for symmetric encryption of files, while RSA is used for "
            "asymmetric key exchange. A strong password turns an encrypted PDF into a "
            "safe container for sensitive reports.",
            "Semantic search understands the meaning of a query instead of matching "
            "keywords exactly. It ranks passages by relevance, so a reworded question "
            "still finds the right paragraph. Vector embeddings power many semantic "
            "search systems in production.",
            "Password managers generate long random passphrases. Never reuse a password "
            "across systems. Multi-factor authentication adds a second factor on top of "
            "the password.",
        ],
    ),
    (
        "Machine Learning Basics",
        [
            "Gradient descent is the workhorse optimization algorithm in machine "
            "learning. It updates model weights by following the negative gradient of "
            "the loss function. Stochastic gradient descent processes one batch at a "
            "time and is much faster on large datasets.",
            "A neural network learns hierarchical representations of data. Deep "
            "networks stack many layers. Transformers use self-attention to capture "
            "long-range dependencies in text.",
        ],
    ),
    (
        "Deep Learning in Practice",
        [
            "Training a deep network requires careful tuning. Gradient descent with "
            "momentum converges faster than vanilla gradient descent on ravines in the "
            "loss landscape. Learning rate schedules and Adam are common choices.",
            "Embeddings map words and documents into a dense vector space where similar "
            "items are close together. Cosine similarity between embeddings is a simple "
            "and effective relevance measure for search.",
        ],
    ),
    (
        "Geography of Rivers",
        [
            "A river bank is the land alongside a river. River bank erosion happens "
            "when fast flowing water undercuts the bank during floods. Willows and "
            "other vegetation stabilize the river bank and reduce erosion.",
            "The Amazon river carries more water than any other river on Earth. Its "
            "basin spans nine countries and holds the largest rainforest.",
        ],
    ),
    (
        "Personal Finance",
        [
            "A bank offers savings accounts, mortgages, and loans. Compare interest "
            "rates before opening an account at a bank. A high-yield savings account at "
            "an online bank often beats a traditional branch.",
            "Diversification means spreading investments across asset classes. An "
            "emergency fund covering three to six months of expenses should sit in a "
            "liquid savings account.",
        ],
    ),
    (
        "Cooking Notes",
        [
            "A good tomato sauce simmers for at least thirty minutes. Garlic, olive "
            "oil, and a pinch of chili flakes form the base. Finish with fresh basil.",
            "Sourdough bread needs a living starter, flour, water, and salt. The bulk "
            "fermentation can take four to twelve hours depending on temperature.",
        ],
    ),
]


def make_plain_pdf(path: Path):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    for title, paras in PAGES:
        y = height - 72
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, y, title)
        y -= 28
        c.setFont("Helvetica", 11)
        for para in paras:
            for line in _wrap(para, 90):
                c.drawString(72, y, line)
                y -= 14
                if y < 72:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y = height - 72
            y -= 10
        c.showPage()
    c.save()


def _wrap(text: str, width: int):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def encrypt(src: Path, dst: Path, user_pw: str, owner_pw: str, algorithm: str):
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password=user_pw, owner_password=owner_pw, algorithm=algorithm)
    with open(dst, "wb") as fh:
        writer.write(fh)


def main():
    plain = OUT / "sample.pdf"
    make_plain_pdf(plain)
    encrypt(plain, OUT / "sample_aes256.pdf", "s3cret-pass", "owner-pw", "AES-256")
    encrypt(plain, OUT / "sample_rc4.pdf", "s3cret-pass", "owner-pw", "RC4-128")
    encrypt(plain, OUT / "sample_owner_only.pdf", "", "owner-pw", "AES-256")
    print("fixtures written to", OUT)
    for f in sorted(OUT.iterdir()):
        print(" ", f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    sys.exit(main())
