
from app.embed import EmbeddingManager
from app.embed import VectorStore
from typing import List, Dict, Any

class RAGRetriever:
    def __init__(self,embedding_manager: EmbeddingManager):
        self.vector_store = None
        self.embedding_manager = embedding_manager

    def retrieve(self, store: VectorStore, query: str, top_k: int = 5, score_threshold: float = -1.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query."""
        self.vector_store = store  # Initialize the vector store with the appropriate collection based on the doc_type
        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            print(f"collection name: {self.vector_store.collection.name}")
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs = []

            if results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Converting the distance to similarity score because chromadb uses cosine distance
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:  # Filter out documents with zero similarity
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "distance": distance,
                            "similarity_score": similarity_score,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents for the query: '{query}'")
            else:
                print(f"No documents found for the query: '{query}'")

            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        
