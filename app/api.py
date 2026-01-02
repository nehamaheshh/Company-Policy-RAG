from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.ingest import ingest_pdf
from app.rag import answer_question

app = FastAPI(title="Company Policy LLM Bot")

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

@app.post("/chat")
async def chat(company_id: str, question: str):
    answer = answer_question(company_id, question)
    return {"answer": answer}
