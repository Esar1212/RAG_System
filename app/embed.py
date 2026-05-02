import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import os


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """ Hugging face model is used over here """
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """ Load the sentence transformer model for generating embeddings. """
        try:
          self.model = SentenceTransformer(self.model_name)
          print(f"Model loaded successfully... Embedding dimension : {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
          print(f"Error loading model: {e}")
          raise e
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """ Generate embeddings for a list of texts. """
        try:
          embeddings = self.model.encode(texts, show_progress_bar=True)
          print(f"Generated embeddings with shape: {embeddings.shape}")
          return embeddings
        except Exception as e:
          print(f"Error generating embeddings: {e}")
          raise e



class VectorStore:
    def __init__(self, collection_name, persist_directory: str = "../data/chromadb"):
        """Initialize the ChromaDB client and create a collection for storing document embeddings."""
        self.client = None
        self.collection = None
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Initialize the ChromaDB client and create a collection for storing document embeddings."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={f"description": "Collection for storing {self.collection_name} embeddings"},
            )

            print(f"ChromaDB collection '{self.collection_name}' initialized successfully.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise e
    
    def get_collection_info(self):
        """Get information about the ChromaDB collection."""
        
        print(f"Collection count: {self.collection.count()}")
        collections = self.client.list_collections()

        for col in collections:
           print(col.name, col.id)


    def add_documents(self, documents: List[Any], embeddings: np.ndarray): 
        """
        Add docs and their corresponding embeddings to the ChromaDB collection.
        Args:
            documents (List[Any]): List of documents to be added.
            embeddings (np.ndarray): Corresponding embeddings for the documents.
        """   

        if(len(documents) != len(embeddings)):
            raise ValueError("The number of documents and embeddings must be the same.")
        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Preparing the metadata
            metadata = dict(doc.metadata)  # Copying the existing metadata from the document
            metadata['doc_idx'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding list
            embeddings_list.append(embedding.tolist())  # Converting numpy array to list for JSON serialization
         
        # Adding documents to the ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                documents=documents_text,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            print(f"Successfully added {len(documents)} documents to the vector store.")
            print(f"Total documents in the collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error adding documents to the vector store: {e}")
            raise e
        

