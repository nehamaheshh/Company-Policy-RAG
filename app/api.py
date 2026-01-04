from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.rag import answer_question

app = FastAPI(title="Company Policy LLM Bot")

@app.get("/")
def home():
    return {"status": "ok", "message": "Go to /docs"}

@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(
    company_id: str = Form(...),
    doc_name: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    pdf_path = f"data/raw/{company_id}_{doc_name}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(contents)
    ingest_pdf(company_id, doc_name, pdf_path)
    return {"status": "ok", "message": "PDF ingested successfully."}



class ChatRequest(BaseModel):
    company_id: str
    question: str



@app.post("/chat")
async def chat(req: ChatRequest):
    return answer_question(req.company_id, req.question, return_sources=True)