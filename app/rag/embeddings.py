from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, VECTOR_DIR

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def store_dir(session_id):
    path = VECTOR_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def create_vectorstore(docs, session_id):
    return Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=store_dir(session_id)
    )


def load_vectorstore(session_id):
    return Chroma(
        embedding_function=embeddings, persist_directory=store_dir(session_id)
    )
