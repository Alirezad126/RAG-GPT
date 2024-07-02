from langchain_openai import ChatOpenAI
from RAGModel.promptTemplate import create_rag_chain
from RAGModel.embeddingModel import load_embedding_vectordb
from langchain_core.messages import HumanMessage, AIMessage

def load_rag_chain(session_id):
    db = load_embedding_vectordb(session_id)  # Loading session-specific VectorDB
    retriever = db.as_retriever(search_kwargs={'k': 8})  # VectorDB as the retriever for model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)  # LLM Model 
    return create_rag_chain(llm, retriever)  # Create RAG chain instance

def get_completion(conversation):
    session_id = conversation.session_id  # Assuming session_id is part of the conversation object
    rag_chain = load_rag_chain(session_id)
    chat_history = []
    question = conversation.message
    history = conversation.conversationState
    print(history)
    
    for msg in history:
        if msg.type == "user":
            chat_history.append(HumanMessage(content=msg.message))
        else:
            chat_history.append(AIMessage(content=msg.message))
    
    result = rag_chain.invoke({"input": question, "chat_history": chat_history})["answer"]
    return result
