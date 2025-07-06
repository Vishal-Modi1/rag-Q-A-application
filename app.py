import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
# No need for HuggingFaceEmbeddings if not used
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
# Corrected: OpenAIEmbeddings requires OPENAI_API_KEY, not GROQ_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM with your Groq API key
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

# Define the ChatPromptTemplate
# IMPORTANT: Use {input} for the user's question as create_stuff_documents_chain expects it
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Function to create or load FAISS vector store
def create_vector_embedding():
    if "vectors" not in st.session_state:

            st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            st.write("Reading uploaded PDF(s)...")
            st.session_state.loader = PyPDFDirectoryLoader("documents")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.raw_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)

            st.write("Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(st.session_state.raw_docs)

            st.write("Creating vector DB...")
            
            st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)

            st.write("Saving vector DB to disk...")
            st.session_state.vectors.save_local("faiss_index")

            st.write("✅ Vector DB is ready.")
    else:
        st.info("✅ Embedding already exists in session")

# Streamlit UI elements
st.title("RAG Document Q&A App")

user_prompt = st.text_input("Enter your query from the document:")

# Button to trigger document embedding
if st.button("Generate Document Embeddings"):
    create_vector_embedding()

import time

# Process user query if embeddings are ready and a prompt is entered
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please generate document embeddings first by clicking 'Generate Document Embeddings'.")
    else:
        # Create the document chain
        doc_chain = create_stuff_documents_chain(llm, prompt=prompt)
        
        # Get the retriever from the FAISS vector store
        retriever = st.session_state.vectors.as_retriever()

        start = time.process_time()

        # Construct the RAG chain using RunnableParallel and RunnablePassthrough
        # IMPORTANT: Ensure the output keys match the prompt's expected variables ({context} and {input})
        retriever_chain = RunnableParallel(
            context=retriever,          # The retriever will be invoked with the input and its output becomes 'context'
            input=RunnablePassthrough() # The original input (user_prompt) is passed through as 'input'
        ) | doc_chain # The output of RunnableParallel is fed into the doc_chain

        # Invoke the entire chain with the user's prompt (string)
        response = retriever_chain.invoke(user_prompt)
        
        print(f"Response time: {time.process_time() - start:.2f} seconds")

        # Display the answer
        # The create_stuff_documents_chain returns a string directly, not a dictionary with "answer" key
        st.write(response)

