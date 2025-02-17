from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
import pandas as pd
import streamlit as st
st.title(" A file Analyzer")
#load api and embeddings
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
groq_API="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"
llm=ChatGroq(groq_api_key=groq_API,model="Llama3-8b-8192")
upload_file=st.file_uploader("upload your file")
if upload_file is not None:
    extension=upload_file.name.split(".")[-1].lower()
    file_path=f"file{extension}"
    with open(file_path,"wb")as file:
        file.write(upload_file.getbuffer())
    st.success(f"file {upload_file.name}saved successfully")
    prompt = ChatPromptTemplate.from_template(
        "You are an expert research assistant. Use the provided context to answer the query.\n"
        "If unsure, state that you don't know. Be concise and factual.\n"
        "Context: {document_context}\n"
        "Query: {user_query}\n"
        "Answer:"
    )

    parser = StrOutputParser()

    # Function to Process PDF
    def process_pdf(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # FAISS Vector Store
        from langchain.vectorstores import Chroma
        vector = Chroma.from_documents(split_docs, embed)
        retriever = vector.as_retriever()

        # Create Processing Chain
        chain = (
            RunnableParallel({"document_context": retriever, "user_query": RunnablePassthrough()})
            | prompt
            | llm
            | parser
        )
        return chain

    # Function to Process Excel
    def process_excel(file_path):
        df = pd.read_excel(file_path)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, handle_parsing_errors=True,allow_dangerous_code=True)
        return agent

    # Function to Process CSV
    def process_csv(file_path):
        df = pd.read_csv(file_path)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, handle_parsing_errors=True,allow_dangerous_code=True)
        return agent

    # User Query Input
    user_query = st.text_input("Enter your query")

    if user_query:
        if extension == "pdf":
            chain = process_pdf(file_path)
            response = chain.invoke(user_query)
            st.write(response)

        elif extension in ["xls", "xlsx"]:
            agent = process_excel(file_path)
            response = agent.invoke(user_query)
            st.write(response)

        elif extension == "csv":
            agent = process_csv(file_path)
            response = agent.invoke(user_query)
            st.write(response)

        else:
            st.error("Unsupported file format. Please upload a PDF, CSV, or Excel file.")