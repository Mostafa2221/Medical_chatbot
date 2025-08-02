import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="Vectorstore/dbfaiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

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


def load_llm():
    llm=ChatGroq(
                    model="deepseek-r1-distill-llama-70b",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=os.environ["API_KEY"]
                )
    return llm
def format_chat_history(history):
    return "\n".join([f"User: {msg['user']}\nMediBot: {msg['bot']}" for msg in history])


def main():
    st.title("Ask Chatbot!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        
        try: 
            vectorstore=get_vectorstore()
        
            if vectorstore is None:
                st.error("Failed to load the vector store")
            
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm(),
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": set_custom_prompt()}
                )
                # Format history for the prompt
            formatted_history = format_chat_history(st.session_state.chat_history)
            
            response = qa_chain.invoke({
                "question": prompt,
                "chat_history": formatted_history
            })

            result=response["answer"]
              # Save the turn in chat history and UI
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            st.session_state.chat_history.append({"user": prompt, "bot": result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
