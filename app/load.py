from fastapi import FastAPI, File, UploadFile, Form
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Used for splitting the documents into smaller chunks
from langchain_experimental.text_splitter import SemanticChunker # Used for splitting the documents into smaller chunks based on semantic meaning
from langchain_community.embeddings import HuggingFaceEmbeddings # Used as an embedding interface to be fed to Semantic Chunker for generating embeddings for the semantic chunking
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import os

class DocumentProcessor:
    def __init__(
        self,
        data_path: str = "data/PDF_files",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.embedding_interface = HuggingFaceEmbeddings(model_name=self.embedding_model)

    def load_documents(self,doc_name:str):
        """Load a single PDF document specified by doc_name"""

        # Ensure .pdf extension
        if not doc_name.endswith(".pdf"):
            doc_name += ".pdf"

        file_path = os.path.join(self.data_path, doc_name)

        # Validate file existence
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Document '{doc_name}' not found in {self.data_path}")

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        return documents
        # loader = DirectoryLoader(
        #     self.data_path,
        #     glob="**/*.pdf",
        #     loader_cls=PyMuPDFLoader,
        #     show_progress=False,
        # )
        # docs = loader.load()
        # return docs

    def recursive_chunking(
        self,
        docs: List,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Perform recursive chunking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def semantic_chunking(self, docs: List):
        """Perform semantic chunking"""
        splitter = SemanticChunker(
            self.embedding_interface,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def process_documents(self,doc_name:str):
       
        docs = self.load_documents(doc_name)

        pre_chunks = self.recursive_chunking(
            docs,
            chunk_size=2000,
            chunk_overlap=200
        )

        final_chunks = self.semantic_chunking(pre_chunks)

        print(f"Loaded {len(docs)} documents")
        print(f"Final chunks after processing: {len(final_chunks)}")

        return final_chunks