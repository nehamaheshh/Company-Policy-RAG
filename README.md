# 🏢 Company Policy Chatbot (Local RAG System)

A **local, privacy-preserving company policy chatbot** built with **FastAPI, ChromaDB, and Ollama**.  
Organizations can upload internal policy documents (PDFs), and employees can ask questions that are answered **strictly based on the uploaded policies**, with **grounded citations** and **safe refusals** when content is not present.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that runs **entirely on-prem / locally**, making it suitable for sensitive enterprise data (HR, Legal, Compliance).

---

## ✨ Key Features

- 📄 Upload policy PDFs via API (supports multi-company ingestion)
- 🔍 Semantic retrieval using embeddings + ChromaDB
- 🎯 **Reranking** for higher answer precision (retrieve wide → rerank smart → answer narrow)
- 🧠 Local LLM inference via Ollama (no cloud APIs)
- 🧷 Grounded answers with source metadata
- 🚫 Safe refusals when a policy is not explicitly covered
- 🖥️ Streamlit frontend for upload + chat
- ⚙️ Modular backend (easy to extend: auth, OCR, versioning)

---

## 🧠 High-Level Architecture

PDF Upload
↓
Save raw PDF (audit & reproducibility)
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
ChromaDB (Vector Store)
↓
User Question
↓
Question Embedding
↓
Retrieve N candidates
↓
Rerank → Top K evidence
↓
Prompt Construction (grounded)
↓
Local LLM (Ollama / LLaMA 3)
↓
Answer + Sources


---

## 🧩 Tech Stack

| Component | Technology |
|---------|------------|
| Backend API | FastAPI |
| Vector Database | ChromaDB (persistent) |
| Embeddings | SentenceTransformers (BGE) |
| Reranker | Cross-Encoder (MS MARCO MiniLM) |
| LLM Runtime | Ollama |
| LLM Model | LLaMA 3 (8B) |
| PDF Parsing | pypdf |
| Frontend | Streamlit |
| OS | Windows (tested), portable |

---

## 📁 Project Structure

company-llm-bot/
├── app/
│ ├── api.py # FastAPI endpoints (/ingest/pdf, /chat)
│ ├── ingest.py # PDF ingestion pipeline
│ ├── rag.py # Retrieval + reranking + prompt construction
│ ├── llm_client.py # Ollama client
│ ├── config.py # Config & paths
│ └── init.py
├── frontend/
│ └── app.py # Streamlit UI (upload + chat)
├── data/
│ ├── raw/ # Uploaded policy PDFs (ignored in git)
│ └── chroma/ # ChromaDB storage (ignored in git)
├── requirements.txt
└── README.md


---

## 🔄 End-to-End Workflow

### 1️⃣ Policy Ingestion (`POST /ingest/pdf`)
1. Admin uploads a policy PDF  
2. PDF is saved to `data/raw/` for auditability  
3. Text is extracted page-by-page  
4. Text is chunked with overlap  
5. Each chunk is embedded into vectors  
6. Chunks + metadata are stored in ChromaDB  

### 2️⃣ Question Answering (`POST /chat`)
1. User submits a question  
2. Question is embedded into a vector  
3. ChromaDB retrieves **N candidate chunks**  
4. Candidates are **reranked** and top **K chunks** are selected  
5. A grounded prompt is constructed  
6. Ollama generates a response locally  
7. Answer is returned with sources and confidence  

---

## 🧠 Why RAG Over Fine-Tuning?

Fine-tuning is not ideal for internal policy systems where content changes frequently and auditability matters.

**RAG was chosen because:**

- 📄 **Instant updates** via document re-indexing
- 🚫 **Lower hallucination risk**
- 🔍 **Traceable answers** with document sources
- 🔄 **Model flexibility** without retraining
- 🔐 **On-prem privacy** for sensitive data

> **Fine-tuning teaches a model how to speak.  
> RAG teaches a system what to know — safely.**

---

## 🚀 Setup (Windows)

### 1️⃣ Create & activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1



### 2️⃣ Install dependencies
pip install -r requirements.txt


### 3️⃣ Install Ollama & pull model
ollama pull llama3:8b


### ▶️ Run Backend (FastAPI)
python -m uvicorn app.api:app --reload
Swagger UI:
http://127.0.0.1:8000/docs

### 🖥️ Run Frontend (Streamlit)
python -m streamlit run frontend/app.py
UI:
http://localhost:8501
```

📌 Current Limitations:

OCR not enabled for scanned PDFs
Authentication & role-based access not implemented
Page/section-level citations pending
No automated evaluation suite yet



