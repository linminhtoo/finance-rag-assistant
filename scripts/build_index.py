import json
import pickle
from pathlib import Path
from typing import Generator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from tqdm import tqdm

# EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast baseline
EMB_MODEL = "BAAI/bge-small-en-v1.5"  # upgrade


def load_records(path: Path) -> Generator[dict[str, str | dict[str, str]], None, None]:
    for line in tqdm(Path(path).open("r", encoding="utf-8"), desc="loading processed records"):
        yield json.loads(line)


def main():
    recs = list(load_records("data/processed.jsonl"))
    logger.success(f"Loaded {len(recs)} records")

    docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in tqdm(recs, desc="creating Document objects")]
    logger.success(f"Created {len(docs)} Document objects")

    emb = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # BGE prefers normalized vectors
    )
    vs = FAISS.from_documents(docs, emb)
    logger.success("Built FAISS index")

    vs.save_local("artifacts/faiss_index")
    with open("artifacts/emb_model_bge.pkl", "wb") as f:
        pickle.dump(EMB_MODEL, f)
    logger.success("Saved FAISS index and embedding model info")


if __name__ == "__main__":
    main()

    """
    (finance-rag-assistant-py3.12) mlin@gpu6:~/repos/scratch/finance-rag-assistant/scripts$ python3 build_index.py
    loading processed records: 5327it [00:00, 60567.85it/s]
    2025-10-22 01:58:29.448 | SUCCESS  | __main__:main:23 - Loaded 5327 records
    creating Document objects: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5327/5327 [00:00<00:00, 37537.41it/s]
    2025-10-22 01:58:29.591 | SUCCESS  | __main__:main:26 - Created 5327 Document objects
    modules.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<00:00, 4.57MB/s]
    config_sentence_transformers.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 1.88MB/s]
    README.md: 94.8kB [00:00, 138MB/s]
    sentence_bert_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52.0/52.0 [00:00<00:00, 602kB/s]
    config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 743/743 [00:00<00:00, 4.67MB/s]
    model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 133M/133M [00:02<00:00, 57.8MB/s]
    tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:00<00:00, 5.29MB/s]
    vocab.txt: 232kB [00:00, 47.3MB/s]
    tokenizer.json: 711kB [00:00, 119MB/s]
    special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [00:00<00:00, 2.05MB/s]
    config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 1.07MB/s]
    2025-10-22 01:58:51.792 | SUCCESS  | __main__:main:31 - Built FAISS index
    2025-10-22 01:58:51.864 | SUCCESS  | __main__:main:36 - Saved FAISS index and embedding model info
    """
