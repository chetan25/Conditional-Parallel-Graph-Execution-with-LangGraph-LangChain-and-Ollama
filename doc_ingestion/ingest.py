import os
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

FILE_PATH = "./doc_ingestion/earthqk.pdf"
# print(os.path.isfile(FILE_PATH), "path")
# print(os.getcwd())

# loader = UnstructuredPDFLoader("./earth.pdf")

loader = PyPDFLoader(
    file_path = FILE_PATH,
    extract_images = True,
    # headers = None
    # extraction_mode = "plain",
    # extraction_kwargs = None,
)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

doc_splits = loader.load_and_split(text_splitter=text_splitter)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-embds",
    embedding=embeddings,
    # persist_directory="./.chroma"
)

retriever = vectorstore.as_retriever()
