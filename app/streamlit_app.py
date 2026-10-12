"""Streamlit chat UI for DocuMind. Talks to the FastAPI backend."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
st.title("📄 DocuMind: Ask your documents")
st.caption("Retrieval-Augmented Generation with cited answers. Upload PDFs/TXT/MD, then ask questions.")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a file", type=["pdf", "txt", "md"])
    if uploaded and st.button("Index file"):
        with st.spinner("Chunking and embedding..."):
            r = requests.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=300,
            )
        if r.ok:
            st.success(f"Indexed {r.json()['chunks_added']} chunks from {uploaded.name}")
        else:
            st.error(r.json().get("detail", r.text))
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()
        for name, n in docs.items():
            st.write(f"• **{name}** ({n} chunks)")
        if not docs:
            st.info("No documents indexed yet.")
    except requests.RequestException:
        st.error(f"Cannot reach API at {API_URL}")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, 4)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for s in turn.get("sources", []):
            with st.expander(f"[{s['id']}] {s['source']}, page {s['page']} (score {s['score']})"):
                st.write(s["text"])

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            r = requests.post(f"{API_URL}/ask", json={"question": q, "top_k": top_k}, timeout=120)
        if r.ok:
            data = r.json()
            st.markdown(data["answer"])
            st.caption(f"{data['latency_ms']} ms" + (" · refused (not in documents)" if data["refused"] else ""))
            for s in data["sources"]:
                with st.expander(f"[{s['id']}] {s['source']}, page {s['page']} (score {s['score']})"):
                    st.write(s["text"])
            st.session_state.history.append(
                {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
            )
        else:
            st.error(r.json().get("detail", r.text))
