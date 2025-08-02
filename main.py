import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from dotenv import load_dotenv, find_dotenv
import re

load_dotenv(find_dotenv())
def clean_response(response_text):
    """Remove thinking blocks and other unwanted patterns from the response"""
    # Remove <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining thinking patterns
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned.strip())
    
    return cleaned.strip()


DB_FAISS_PATH = "Vectorstore/dbfaiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.0,
        groq_api_key=os.environ["API_KEY"]
    )

def set_custom_prompt():
    template = """You are MediBot, a helpful, accurate, and cautious AI medical assistant.

Your task is to analyze the user's question along with the given medical context and prior conversation. Provide **possible medical explanations or conditions** only when clearly supported by the context.

Always follow these principles:
- ❌ Do **not guess or speculate**.
- ✅ Only make inferences if the **context explicitly supports** it.
- 🧠 Use **estimated probabilities** where medically appropriate.
- 📚 Stick strictly to the medical information provided.
- 👨‍⚕️ Always **remind the user to consult a licensed medical professional** at the end.

---

📄 **Medical Context**:
{context}

💬 **Conversation History**:
{chat_history}

❓ **User Question**:
{question}

"""
    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )

def initialize_qa_chain():
    """Initialize the QA chain with persistent memory"""
    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        
        # Initialize memory that persists across requests
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": set_custom_prompt()},
            return_source_documents=True  # Optional: to see source documents
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Failed to initialize QA chain: {str(e)}")
        return None
    

def main():
    st.title("MediBot - Medical Assistant Chatbot")
    
    # Initialize messages for UI display
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize QA chain
    qa_chain = initialize_qa_chain()
    if qa_chain is None:
        st.error("Failed to initialize the chatbot. Please check your configuration.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your medical concerns..."):
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Generate response
        try:
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    # The memory is automatically managed by the chain
                    response = qa_chain.invoke({"question": prompt})
                    raw_result = response["answer"]
                    result = clean_response(raw_result)
                    
                    st.markdown(result)
                    
                    # Optionally display source documents
                    if "source_documents" in response and response["source_documents"]:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content[:500] + "...")
            
            # Save assistant message
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # You might want to log this error for debugging

if __name__ == "__main__":
    main()
