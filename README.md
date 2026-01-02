# 🏢 Company Policy Chatbot (Local RAG System)

A **local, privacy-preserving company policy chatbot** built using **FastAPI, ChromaDB, and Ollama**.  
Organizations can upload internal policy documents (PDFs), and employees can ask questions that are answered **strictly based on the uploaded policies**, with **no hallucinations**.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that runs **entirely on-prem / locally**, making it suitable for sensitive enterprise data (HR, Legal, Compliance).

---

## ✨ Key Features

- 📄 Upload company policy PDFs (HR, compliance, employee handbooks)
- 🔍 Semantic search over policies using vector embeddings
- 🧠 Local LLM inference via Ollama (no cloud APIs)
- 🧷 Grounded answers with source references
- 🚫 Safe refusals when a policy is not explicitly covered
- 🏢 Multi-company support via metadata filtering
- ⚙️ Modular, extensible backend architecture

---

## 🧠 High-Level Architecture

```
PDF Upload
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
Semantic Retrieval + Reranking
   ↓
Prompt Construction
   ↓
Local LLM (Ollama / LLaMA 3)
   ↓
Answer + Citations
```

---

## 🧩 Tech Stack

| Component | Technology |
|---------|------------|
| API | FastAPI |
| Vector Database | ChromaDB |
| Embeddings | SentenceTransformers (BGE) |
| LLM Runtime | Ollama |
| LLM Model | LLaMA 3 (8B) |
| PDF Parsing | pypdf |
| Language | Python |
| OS | Windows (tested), portable |

---

## 📁 Project Structure

```
company-llm-bot/
├── app/
│   ├── api.py          # FastAPI endpoints
│   ├── ingest.py       # PDF ingestion pipeline
│   ├── rag.py          # Retrieval + RAG logic
│   ├── llm_client.py   # Ollama client
│   ├── config.py       # Config & paths
│   └── __init__.py
├── data/
│   ├── raw/            # Uploaded policy PDFs (source of truth)
│   └── chroma/         # ChromaDB persistent storage
├── requirements.txt
└── README.md
```

---

## 🔄 End-to-End Workflow

### 1️⃣ Policy Ingestion (`POST /ingest/pdf`)
1. Admin uploads a policy PDF  
2. PDF is saved to `data/raw/` (audit & reproducibility)  
3. Text is extracted page-by-page  
4. Text is chunked with overlap  
5. Each chunk is embedded into vectors  
6. Chunks + metadata are stored in ChromaDB  

### 2️⃣ Question Answering (`POST /chat`)
1. User submits a question  
2. Question is embedded into a vector  
3. ChromaDB retrieves top-N relevant chunks  
4. Retrieved chunks are **reranked** for precision  
5. A grounded prompt is built  
6. Ollama runs the local LLM  
7. Answer is returned with citations or a safe refusal  

---

## 🛡️ Safety & Reliability

- ❌ No hallucinated answers  
- ✅ Answers strictly use retrieved policy text  
- ⚠️ Explicit “not found” responses when policies don’t exist  
- 🔍 Source chunk references included  
- 🔐 All data stays local  

This behavior is **intentional** and **enterprise-safe**.

---

## 🧠 Why RAG Over Fine-Tuning?

Fine-tuning is not well-suited for internal policy systems.

**Retrieval-Augmented Generation (RAG)** was chosen because:

- 📄 **Policies change frequently** → RAG updates instantly by re-indexing documents  
- 🚫 **Hallucination prevention** → LLM only sees retrieved policy text  
- 🔍 **Auditability** → answers are traceable to specific document chunks  
- 🔄 **Model flexibility** → swap LLMs without retraining  
- 🔐 **On-prem deployment** → no sensitive data sent to cloud APIs  

> **Fine-tuning teaches a model how to speak.  
> RAG teaches a system what to know — safely.**

---

## 🚀 Setup Instructions

### 1️⃣ Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2️⃣ Install dependencies
```powershell
pip install -r requirements.txt
```

### 3️⃣ Install Ollama & pull model
Download Ollama from: https://ollama.com/download

```powershell
ollama pull llama3:8b
```

### 4️⃣ Run the API
```powershell
python -m uvicorn app.api:app --reload
```

Open:
👉 http://127.0.0.1:8000/docs

---

## 🧪 Example Usage

### Upload a policy PDF
`POST /ingest/pdf`

- `company_id`: `acme`
- `doc_name`: `handbook_2025`
- `file`: employee handbook PDF

### Ask a question
`POST /chat`

```json
{
  "company_id": "acme",
  "question": "What is the sick leave policy?"
}
```

---

## 📌 Current Limitations

- OCR not enabled for scanned PDFs  
- Authentication & role-based access not yet implemented  
- Page/section-level citations pending  
- No frontend UI (API-only)  

---

## 🔜 Planned Improvements

- 🔐 JWT authentication & role-based access  
- 📑 Page + section citations  
- 🧠 Hybrid retrieval (semantic + keyword)  
- 🧪 Evaluation harness for answer quality  
- 🖥️ Web UI (Streamlit / React)  
- 📦 Dockerized deployment  
- 📚 Document versioning & lifecycle management  

---
