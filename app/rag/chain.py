from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, TOP_K
from app.rag.embeddings import load_vectorstore
from app.rag.prompts import create_rag_chain


def load_chain(session_id):
    retriever = load_vectorstore(session_id).as_retriever(
        search_kwargs={"k": TOP_K}
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.5)
    return create_rag_chain(llm, retriever)


def answer(session_id, question, history):
    """Answer `question` against the session's document, given prior turns."""
    chat_history = [
        HumanMessage(content=turn.message)
        if turn.type == "user"
        else AIMessage(content=turn.message)
        for turn in history
    ]
    chain = load_chain(session_id)
    return chain.invoke({"input": question, "chat_history": chat_history})["answer"]
