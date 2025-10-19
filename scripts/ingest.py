"""
Goals:
- Robust PDF -> text (fallbacks).
- Sensible chunking: ~700–1000 tokens with ~150–200 overlap (or sentence-aware).
- Store metadata for filtering & citations.
"""
import json
import re
from pathlib import Path

# from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed.jsonl")

def load_docs():
    docs = []
    for p in RAW_DIR.rglob("*"):
        if p.suffix.lower() in [".pdf"]:
            try:
                loader = PyPDFLoader(str(p))
                docs.extend(loader.load())
            except Exception:
                pass
        elif p.suffix.lower() in [".md", ".txt", ".html"]:
            docs.append(TextLoader(str(p), encoding="utf-8").load()[0])
    return docs

def clean_text(t: str) -> str:
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def main():
    raw_docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3200, chunk_overlap=400,  # characters ~ ~800–1000 tokens
        separators=["\n\n", "\n", " ", ""],
    )
    records = []
    for d in raw_docs:
        meta = {**(d.metadata or {})}
        meta["source"] = meta.get("source") or meta.get("file_path")
        for chunk in splitter.split_text(clean_text(d.page_content)):
            records.append({"text": chunk, "metadata": meta})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
