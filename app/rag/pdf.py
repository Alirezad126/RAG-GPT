from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.chunking import split_text


def chunk_pdf(data):
    """Read PDF bytes and split the text into overlapping chunks."""
    reader = PdfReader(data)
    raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)

    if not raw_text.strip():
        raise ValueError(
            "No extractable text found in the PDF — it may be a scanned image."
        )

    return split_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
