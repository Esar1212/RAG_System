import streamlit as st
import os
from app.embed import EmbeddingManager, VectorStore
from app.load import DocumentProcessor  
from app.retrieve import RAGRetriever
from app.generate import RGenerator

st.set_page_config(page_title="RAG Q&A System", layout="centered")

st.title("RAG Document Q&A System")
st.markdown("Ask questions from your PDF documents using AI")

UPLOAD_DIR = "data/PDF_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------- STEP 1: Ask for document name --------
doc_name_input = st.text_input("Enter document name (without .pdf):")

doc_name = None
file_path = None

if doc_name_input:
    doc_name = doc_name_input.strip()
    file_path = os.path.join(UPLOAD_DIR, f"{doc_name}.pdf")

    # -------- STEP 2: Check if file exists --------
    if os.path.exists(file_path):
        st.success(f"Document '{doc_name}.pdf' found. No upload needed.")
    else:
        st.warning(f"Document '{doc_name}.pdf' not found. Please upload it.")

        uploaded_file = st.file_uploader("Upload the PDF file", type=["pdf"])

        if uploaded_file:
            # Save file with the entered doc_name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Uploaded and saved as {doc_name}.pdf")


# -------- INITIALIZATION --------
@st.cache_resource
def initialize_system(doc_name):
    logs = []

    embedder = EmbeddingManager()
    logs.append("Embedding model loaded")

    store = VectorStore(doc_name)
    logs.append(f"Vector store initialized: {doc_name}")

    data = store.collection.get()
    doc_count = len(data["documents"])

    try:
        store.client.get_collection(name=doc_name)
        exists = True
    except:
        exists = False

    # -------- INGESTION --------
    if not exists or doc_count == 0:
        logs.append("No existing data found. Starting ingestion...")

        processor = DocumentProcessor()
        docs = processor.process_documents(doc_name)
        logs.append(f"Loaded {len(docs)} document chunks")

        embeddings = embedder.generate_embeddings(
            [doc.page_content for doc in docs]
        )
        logs.append("Embeddings generated")

        store.add_documents(docs, embeddings)
        logs.append("Documents stored in vector DB")

    else:
        logs.append(f"Existing collection found with {doc_count} documents")

    return embedder, store, logs


# -------- MAIN FLOW --------
if doc_name and os.path.exists(file_path):
    embedder, store, logs = initialize_system(doc_name)

    # Show logs
    with st.expander("Ingestion Logs"):
        for log in logs:
            st.write(log)

    st.info(f"Documents in collection: {store.collection.count()}")

    retriever = RAGRetriever(embedder)
    generator = RGenerator()

    # -------- CHAT MEMORY --------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question:")

    col1, col2 = st.columns([1,1])

    with col1:
        submit = st.button("Submit")
    with col2:
        clear = st.button("Clear Chat")

    if clear:
        st.session_state.chat_history = []

    if submit and query:
        with st.spinner("Thinking..."):
            try:
                retrieved_docs = retriever.retrieve(store, query)

                if not retrieved_docs:
                    answer = "No relevant documents found."
                else:
                    answer = generator.generate_answer(
                        query,
                        [doc["content"] for doc in retrieved_docs]
                    )

                st.session_state.chat_history.append((query, answer))

            except Exception as e:
                st.error(f"Error: {e}")

    # -------- DISPLAY CHAT --------
    if st.session_state.chat_history:
        st.subheader("Conversation")

        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}: {q}**")
            st.write(a)
            st.divider()