"""
Create 10-30 eval Qs:
- Simple factual (“What does X section say about Y?”)
- Multi-hop (“Contrast A vs B across docs”)
- Edge case (“When does the doc not support a claim?”)
- Auto-generate candidates and then prune manually:

# scripts/make_eval.py (sketch)
# Take titles/sections → synthesize Qs using your LLM, then manually review.

-----------------------------------
Score with Ragas (LLM-assisted) or simple string-match baselines.

Track:
- Answer correctness
- Groundedness (uses provided context)
- Context precision (are retrieved chunks relevant?)
- If precision is low → increase rerank top_n, reduce chunk size, add hybrid search (BM25 + vectors). For BM25 you can do: retrieve 10 via BM25 (rank-bm25) + 10 via vectors → union → rerank.

-----------------------------------
Observability -> look into `langfuse` or MLFlow for traces

-----------------------------------
Troubleshooting cheat-sheet
- Bad answers / hallucinations → lower temperature; add “must cite” instruction; add verification pass; improve chunking & rerank.
- Irrelevant context → smaller chunks (e.g., 600-800 tokens), larger overlap (150-200), hybrid retrieval, reranker top-k 50→rerank→top-6.
- Latency → reduce k; switch to bge-small embeddings; switch from cross-encoder to ColBERT-style later; consider API LLM.
- Ollama OOM → use a smaller/quantized model: llama3.2:3b-instruct or mistral:7b q4_K_M.
"""
