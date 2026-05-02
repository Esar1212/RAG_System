# 📄 RAG Document Q&A System

An AI-powered document assistant that enables users to upload PDFs and ask natural language questions using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📂 Dynamic PDF upload & document selection
- 🧠 Semantic search using vector embeddings (ChromaDB)
- 🤖 LLM-powered answer generation (Groq - LLaMA 3)
- 🔁 Persistent vector storage (no reprocessing needed)
- 💬 Interactive chat-based Q&A interface (Streamlit)
- ⚡ Conditional ingestion (only processes new documents)

---

## 🏗️ Architecture

1. **Document Ingestion**
   - PDF loading using PyMuPDF
   - Recursive + semantic chunking
   - Embedding generation using Sentence Transformers

2. **Vector Storage**
   - ChromaDB used for persistent storage
   - Each document stored as a separate collection

3. **Retrieval**
   - Query embedding generation
   - Top-K semantic search using cosine similarity

4. **Generation**
   - Context-aware response using LLM (Groq API)
   - Strict grounding to retrieved context (no hallucination)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Vector DB**: ChromaDB  
- **LLM**: Groq (LLaMA 3.3 70B)  
- **Document Processing**: PyMuPDF, LangChain  

---

## 📸 Demo

### Example Queries:
- "Explain adjacency list representation"
- "What projects are mentioned in the document?"
- "Summarize the key concepts"

---

## ⚙️ Setup

```bash
uv sync
uv run streamlit run app_ui.py