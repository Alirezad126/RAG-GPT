from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def get_db_dir(session_id):
    script_dir = os.path.dirname(__file__)
    db_dir = os.path.join(script_dir, f'vectorDB/{session_id}')
    os.makedirs(db_dir, exist_ok=True)
    return db_dir

def create_embedding_from_texts(docs, session_id):
    db_dir = get_db_dir(session_id)
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=db_dir)
    return db

def load_embedding_vectordb(session_id):
    db_dir = get_db_dir(session_id)
    db = Chroma(embedding_function=embeddings, persist_directory=db_dir)
    return db
