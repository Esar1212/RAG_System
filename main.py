from app.embed import EmbeddingManager, VectorStore
from app.load import DocumentProcessor  
from app.retrieve import RAGRetriever
from app.generate import RGenerator
    

def main():
    print("Upload your PDF documents to the 'data/PDF_files' directory before running the application.")
    
    doc_name = input("Enter the name of the pdf document you just uploaded: ")
    
    print("RAG Document Q&A System")
    print("Type 'exit' to quit\n")

    # Initialize components
    embedder = EmbeddingManager()
    store = VectorStore(doc_name)  

    # Debug: check how many docs exist
    data = store.collection.get()
    print(f"Documents in collection: {len(data['documents'])}")

    # Check if collection exists properly
    try:
        store.client.get_collection(name=doc_name)
        exists = True
    except:
        exists = False

    # If collection is empty, run ingestion
    if not exists or len(data["documents"]) == 0:
        print("No existing data found. Running ingestion...")

        processor = DocumentProcessor()
        docs = processor.process_documents(doc_name)

        embeddings = embedder.generate_embeddings(
            [doc.page_content for doc in docs]
        )

        store.add_documents(docs, embeddings)

        print("Ingestion completed.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        try:
            # Step 2: Retrieve relevant documents
            retriever = RAGRetriever(embedder)
            retrieved_docs = retriever.retrieve(store, query)

            if not retrieved_docs:
                print("No relevant documents found for the query.")
                continue
            
            # Step 3: Generate response
            generator = RGenerator()
            response = generator.generate_answer(
                query,
                [doc["content"] for doc in retrieved_docs]   # ✅ FIXED
            )

            # Step 4: Print result
            print("\n Answer:")
            print(response)
            print("\n" + "-"*50 + "\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()