"""
Goals:
- Robust PDF -> text (fallbacks).
- Sensible chunking: ~700–1000 tokens with ~150–200 overlap (or sentence-aware).
- Store metadata for filtering & citations.
"""
import json
import re
from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def load_docs(input_folder: Path) -> list[Document]:
    docs = []
    for folder in tqdm(input_folder.iterdir(), desc="loading raw documents"):
        text_path = folder / "text.jsonl"
        if not text_path.exists():
            continue
        loader = JSONLoader(
            file_path=text_path,
            jq_schema=".text",
            text_content=True,
            json_lines=True,
        )
        docs.extend(loader.load())
    return docs


def clean_text(t: str) -> str:
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main():
    RAW_DIR = Path("/home/mlin/repos/scratch/finance-rag-assistant/scripts/data/normalized")
    OUT_PATH = Path("/home/mlin/repos/scratch/finance-rag-assistant/scripts/data/processed.jsonl")

    raw_docs = load_docs(input_folder=RAW_DIR)
    splitter = RecursiveCharacterTextSplitter(
        # TODO: manual inspection did not see overlaps between adjacent chunks, is this a bug???
        chunk_size=3200,
        chunk_overlap=400,  # characters ~ ~800–1000 tokens
        separators=["\n\n", "\n", " ", ""],
    )
    records = []
    for d in tqdm(raw_docs, desc="processing documents"):
        meta = d.metadata
        with open(meta["source"], "r") as f:
            file = json.load(f)
            meta |= {k: file[k] for k in ["doc_id", "ticker", "cik", "filing_date", "source_url"]}
        for chunk in splitter.split_text(clean_text(d.page_content)):
            records.append({"text": chunk, "metadata": meta})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
