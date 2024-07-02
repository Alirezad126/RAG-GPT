from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def process_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    chunks = text_splitter.split_text(raw_text)

    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents
