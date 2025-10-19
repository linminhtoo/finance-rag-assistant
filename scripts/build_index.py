# scripts/build_index.py
import json
import pickle
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast baseline
# upgrade: "BAAI/bge-small-en-v1.5"


def load_records(path):
    for line in Path(path).open("r", encoding="utf-8"):
        yield json.loads(line)


def main():
    recs = list(load_records("data/processed.jsonl"))
    docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in recs]
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL, normalize_embedding=True)
    vs = FAISS.from_documents(docs, emb)
    vs.save_local("artifacts/faiss_index")
    with open("artifacts/emb_model.pkl", "wb") as f:
        pickle.dump(EMB_MODEL, f)


if __name__ == "__main__":
    main()
