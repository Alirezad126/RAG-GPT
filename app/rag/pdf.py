from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def process_pdf(file_path):
    """Read a PDF and split its text into overlapping chunks for embedding."""
    reader = PdfReader(file_path)
    raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)

    if not raw_text.strip():
        raise ValueError("No extractable text found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return [Document(page_content=chunk) for chunk in splitter.split_text(raw_text)]
