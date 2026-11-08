from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL
from app.rag.retrieval import top_chunks

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, which might reference "
    "earlier turns, rewrite it as a standalone question that can be understood "
    "on its own. Do NOT answer it. Return it unchanged if it already stands alone."
)

ANSWER_PROMPT = (
    "You answer questions about the document excerpts below. Decline anything "
    "unrelated to the document. Keep answers to three sentences at most, and say "
    "you don't know rather than guessing.\n\n"
    "Document excerpts:\n{context}"
)

_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.5)
    return _llm


def as_messages(history):
    return [
        HumanMessage(content=turn.message)
        if turn.type == "user"
        else AIMessage(content=turn.message)
        for turn in history
    ]


def standalone_question(question, history):
    """Fold the conversation into the question so retrieval can stand alone."""
    if not history:
        return question
    messages = (
        [SystemMessage(content=CONTEXTUALIZE_PROMPT)]
        + as_messages(history)
        + [HumanMessage(content=question)]
    )
    return llm().invoke(messages).content


def answer(question, history, chunks, vectors):
    query = standalone_question(question, history)
    context = "\n\n".join(top_chunks(query, chunks, vectors))
    messages = (
        [SystemMessage(content=ANSWER_PROMPT.format(context=context))]
        + as_messages(history)
        + [HumanMessage(content=question)]
    )
    return llm().invoke(messages).content
