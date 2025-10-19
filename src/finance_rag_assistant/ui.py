"""
Serving choices & trade-offs

Local (Ollama): zero cost, private, great for demos; 7–8B works on your 3080 with 4-bit quant. Latency ~1–2s/token startup then streams fast.

API (GPT-4o-mini / Claude Haiku / Mistral-Large): better instruction-following & grounding; cheap enough for a demo; simpler than fighting quant edge cases.

Embeddings: local bge-small is fine. If you want best-in-class quickly: text-embedding-3-large (API) or nomic-embed-text-v1 (API). You can swap later by rebuilding FAISS.
"""

import streamlit as st
from rag import make_chain

st.set_page_config(page_title="Domain RAG", layout="wide")
st.title("🔎 Domain RAG Assistant")

if "chain" not in st.session_state:
    st.session_state.chain = make_chain()

q = st.text_input("Ask a question:")
if st.button("Ask") and q:
    out = st.session_state.chain({"query": q})
    st.markdown("### Answer")
    st.write(out["result"])
    st.markdown("### Sources")
    for i, d in enumerate(out["source_documents"]):
        src = d.metadata.get("source", "unknown")
        st.caption(f"{i+1}. {src}")
