import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
st.cache_data.clear()

# Set HuggingFace token
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit title and description
st.title("Conversational RAG")
st.write("GROQ_KEY = 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")
st.write("Upload a PDF to start the conversation")

# API key input
api_key = st.text_input("Enter your GROQ API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    session_id = st.text_input("Session ID", value="Default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    # PDF file uploader
    upload_file = st.file_uploader("Choose a PDF file", type="pdf")

    if upload_file:
        try:
            # Save the uploaded PDF to a temporary file
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(upload_file.getbuffer())

            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(temppdf)
            docs = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create a vector store and retriever
            vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vector_store.as_retriever()

            # Set up prompts and chains
            context_system_prompt = "Consider yourself as an expert summarizer."
            context_prompt = ChatPromptTemplate.from_messages([
                ("system", context_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

            system_prompt = ("Consider yourself as an expert in Q&A. "
                             "Use the retrieved content to answer concisely. {context}")
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # User input for the question
            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write(st.session_state.store)
                st.success(response["answer"])
                st.write("Chat history", session_history.messages)

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
else:
    st.warning("Enter your GROQ API key to continue")
