
# PDF Query Application

This application allows users to upload PDF files, input a query, and receive answers based on the content of the uploaded documents. The application is built using Streamlit for the UI, and it leverages LangChain, FAISS, and Google's Generative AI for the backend processing.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- PyPDF2
- LangChain
- FAISS
- HuggingFace Transformers
- Google Generative AI
- dotenv

You can install the required packages using pip:

```bash
pip install streamlit PyPDF2 langchain faiss-cpu transformers google-generativeai python-dotenv
```

## Setting Up

1. **Clone the Repository**:
   Clone the repository to your local machine.

2. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Create Necessary Directories**:
   Ensure the following directories exist:
   - `./data/`
   - `./vectorstores/db_faiss/`

## Running the Application

To run the application, use the following command:

```bash
streamlit run app.py
```

## Code Overview

### Importing Libraries

The application imports various libraries for handling PDF files, embeddings, text splitting, vector storage, and the Streamlit UI.

```python
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
import os
import requests
from huggingface_hub import configure_http_backend
import streamlit as st
```

### Fixing SSL Issues

The application configures the HTTP backend to fix SSL issues.

```python
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
```

### Setting Up Environment Variables

The application loads environment variables from a `.env` file.

```python
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
```

### Initializing Models and Chains

The application initializes the embedding model, LLM model, and QA chain.

```python
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

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

chain = load_qa_chain(
    model,
    chain_type="stuff",
    prompt=prompt_temp,
    document_variable_name="context",
)

prompt_str = prompt_temp.template
prompt = PromptTemplate(template=prompt_str, input_variables=["history", "context", "question"])
db = FAISS.load_local('./vectorstores/db_faiss/', embeddings_model, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    },
)
```

### Loading Data

The `load_data` function saves the uploaded file to a specified directory and loads the documents using `PyPDFDirectoryLoader`.

```python
def load_data(uploaded_file):
    save_path = os.path.join("./data/", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFDirectoryLoader("./data/")
    docs = loader.load()
    return docs
```

### Splitting Text

The `split_text` function splits the documents into smaller chunks.

```python
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    return texts
```

### Storing in Vector DB

The `store_VDB` function stores the text chunks in a FAISS vector database.

```python
def store_VDB(texts):
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings_model)
    vectorstore.save_local("./vectorstores/db_faiss")
    return vectorstore
```

### Querying the QA Chain

The `query_qa_chain` function queries the QA chain with the user's query.

```python
def query_qa_chain(query):
    response = qa_chain({"query": query})
    return response['result']
```

### Streamlit UI

The Streamlit UI allows users to upload a PDF file, input a query, and get an answer.

```python
st.title("File Upload and Query")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
query = st.text_input("Enter your query")
if st.button("Get Answer"):
    if uploaded_file is not None and query:
        docs = load_data(uploaded_file)
        texts = split_text(docs)
        store_VDB(texts)
        result = query_qa_chain(query)
        st.write("Answer:", result)
    else:
        st.write("Please upload a file and enter a query.")
```

## Conclusion

This application demonstrates how to build a PDF query interface using Streamlit, LangChain, FAISS, and Google's Generative AI. Users can upload PDF files, input queries, and receive answers based on the content of the uploaded documents.

If you have any questions or need further assistance, feel free to reach out to Savi Chopra at savisavi2002@gmail.com.

---

Feel free to customize the README file further to suit your needs! If you have any other questions or need additional assistance, let me know!
