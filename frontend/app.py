import json
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Company Policy Chatbot", page_icon="🏢", layout="wide")

st.title("🏢 Company Policy Chatbot (Local RAG)")
st.caption("Upload a policy PDF, then ask questions grounded strictly in the policy text. Runs locally via ChromaDB + Ollama.")

# -----------------------
# Sidebar: Connection + Upload
# -----------------------
st.sidebar.header("⚙️ Setup")

api_base = st.sidebar.text_input("API Base URL", value=API_BASE)

st.sidebar.divider()
st.sidebar.subheader("📄 Upload Policy PDF")

company_id = st.sidebar.text_input("Company ID", value="acme")
doc_name = st.sidebar.text_input("Document Name", value="handbook_2025")
pdf_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

upload_clicked = st.sidebar.button("Upload & Ingest", type="primary", use_container_width=True)

def safe_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}

if upload_clicked:
    if not pdf_file:
        st.sidebar.error("Please select a PDF file first.")
    elif not company_id.strip() or not doc_name.strip():
        st.sidebar.error("Please fill Company ID and Document Name.")
    else:
        with st.sidebar.status("Uploading & ingesting…", expanded=True) as status:
            try:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                data = {"company_id": company_id.strip(), "doc_name": doc_name.strip()}
                resp = requests.post(f"{api_base}/ingest/pdf", files=files, data=data, timeout=600)
                if resp.status_code == 200:
                    status.update(label="✅ Ingested successfully", state="complete")
                    st.sidebar.success(safe_json(resp).get("message", "Ingested"))
                else:
                    status.update(label="❌ Ingest failed", state="error")
                    st.sidebar.error(safe_json(resp))
            except requests.exceptions.RequestException as e:
                status.update(label="❌ API connection error", state="error")
                st.sidebar.error(str(e))

st.sidebar.divider()
st.sidebar.subheader("🧠 Chat Settings")
top_k = st.sidebar.slider("Top K chunks (reranked)", min_value=3, max_value=10, value=5)
retrieve_n = st.sidebar.slider("Retrieve N candidates", min_value=10, max_value=50, value=25)
show_sources = st.sidebar.checkbox("Show sources", value=True)
show_debug = st.sidebar.checkbox("Show debug JSON", value=False)

# -----------------------
# Session state for chat
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload a policy PDF on the left, then ask a question about it."}
    ]

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

# -----------------------
# Chat display
# -----------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a policy question… (e.g., What is the sick leave policy?)")

def call_chat_api(question: str):
    payload = {
        "company_id": company_id.strip(),
        "question": question,
        # If your rag.py supports these knobs, wire them.
        # Otherwise, you can ignore and remove them.
        "top_k": top_k,
        "retrieve_n": retrieve_n,
    }
    # Your current /chat endpoint only accepts company_id, question.
    # So we send only those to avoid 422 errors.
    payload = {"company_id": company_id.strip(), "question": question}
    return requests.post(f"{api_base}/chat", json=payload, timeout=600)

if prompt:
    if not company_id.strip():
        st.error("Please set Company ID in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking… (retrieving policy context + generating answer)"):
                try:
                    resp = call_chat_api(prompt)
                    if resp.status_code != 200:
                        st.error(f"API error ({resp.status_code}): {safe_json(resp)}")
                    else:
                        data = safe_json(resp)
                        answer = data.get("answer", "")
                        confidence = data.get("confidence")
                        sources = data.get("sources", [])

                        # Render answer
                        st.markdown(answer)

                        # Render confidence
                        if confidence:
                            st.caption(f"Confidence: **{confidence}**")

                        # Render sources
                        if show_sources and sources:
                            with st.expander("Sources used"):
                                for s in sources:
                                    doc = s.get("doc_name")
                                    idx = s.get("chunk_idx")
                                    f = s.get("source_file")
                                    score = s.get("rerank_score")
                                    st.write(f"- **{doc}** | chunk **{idx}** | file: `{f}` | rerank: `{score}`")

                        if show_debug:
                            with st.expander("Debug JSON"):
                                st.code(json.dumps(data, indent=2))

                        # Save
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_sources = sources
                        st.session_state.last_meta = {"confidence": confidence}

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

# -----------------------
# Footer controls
# -----------------------
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. Ask a new question."}]
        st.session_state.last_sources = []
        st.session_state.last_meta = {}
        st.rerun()

with col2:
    st.download_button(
        "⬇️ Download chat",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )

with col3:
    st.caption("Tip: For best results, upload text-based PDFs. Scanned PDFs need OCR (planned improvement).")
