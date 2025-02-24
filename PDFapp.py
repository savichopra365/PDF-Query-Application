from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, TextStreamer, pipeline
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import io
import torch

"CODE SUBMITED BY SAVI CHOPRA -987753980 , savisavi2002@gmail.com"

# fixing the SSL issue
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# setting GEMINI key
import os 
os.environ["GOOGLE_API_KEY"] = ""
# using langchain embeddings model
from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# setting the LLM model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
# PROMPT for LLM
from langchain.prompts import PromptTemplate

prompt_temp = PromptTemplate.from_template(
   """
    You are an intelligent assistant designed to answer questions based on the content of uploaded PDF documents.
    Please provide accurate and helpful answers to the questions asked, using the context provided from the documents.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
)

chain=load_qa_chain(
    model,
    chain_type="stuff",
    prompt=prompt_temp,
    document_variable_name="context",
)

prompt_str = prompt_temp.template
prompt = PromptTemplate(template =prompt_str , input_variables = ["history", "context", "question"])
db = FAISS.load_local('./vectorstores/db_faiss/', embeddings_model,allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
        llm = model,
        chain_type ="stuff",
        retriever = db.as_retriever(search_kwargs = {'k': 2}),
        return_source_documents = True, #explaining the answer
        chain_type_kwargs= {
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),

        },
    )



# loading data
def load_data(uploaded_file):
    # Save the uploaded file to the specified directory
    save_path = os.path.join("./data/", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFDirectoryLoader("./data/")
    docs = loader.load()
    return docs
# splitting text
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 64) # play with these values later
    texts = text_splitter.split_documents(docs)
    return texts
# storing in vector DB
def store_VDB(texts):
    vectorstore = FAISS.from_documents(documents= texts, embedding = embeddings_model)
    vectorstore.save_local("./vectorstores/db_faiss")
    return vectorstore
# defining QA chain for answering query
def query_qa_chain(query):
    response = qa_chain({"query": query})
    return response['result']



# STREAMLIT UI
import streamlit as st
st.title("File Upload and Query ")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
query = st.text_input("Enter your query")
if st.button("Get Answer"):
    if uploaded_file is not None and query:
            # Process the uploaded file
            docs = load_data(uploaded_file)
            texts = split_text(docs)
            store_VDB(texts)

            # Query the QA chain
            result = query_qa_chain(query)
            st.write("Answer:", result)
    else:
            st.write("Please upload a file and enter a query.")



