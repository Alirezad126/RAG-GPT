from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do NOT answer the question; just "
    "reformulate it if needed, and otherwise return it as is."
)

ANSWER_PROMPT = (
    "You answer questions about the document excerpts provided below. Decline "
    "anything unrelated to the document. Keep answers to three sentences at "
    "most, and say you don't know rather than guessing.\n\n"
    "{context}"
)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def create_rag_chain(llm, retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
