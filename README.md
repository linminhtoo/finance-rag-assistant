# Finance RAG Assistant

A Retrieval-Augmented Generation (RAG) system that can answer financial questions using SEC 10-K filings and earnings call transcripts.

---

# TODO: Problem -> Approach -> Architecture -> Results (Evals) -> Lessons -> Future Work
- Host demo: Gradio, Streamlit, or HuggingFace Space
- Record a short Loom video walkthrough

Resume
- Built and deployed a retrieval-augmented generation (RAG) system using OpenAI GPT-4-turbo and FAISS for document QA over proprietary energy data.
- Designed an LLM agent capable of autonomous multi-step reasoning and tool use via LangGraph.
- Implemented parameter-efficient fine-tuning (LoRA) of Mistral-7B on domain-specific customer support transcripts to improve response tone alignment.

Knowledge
- Embeddings: cosine similarity, dimensionality, vector DB indexing
- Prompting patterns: few-shot, chain-of-thought, ReAct
- Evaluation: hallucination detection, grounding metrics (ragas, FaithEval)
- Post-training: LoRA, quantization, adapters, RLHF overview
- Tooling: LangChain, LlamaIndex, DSPy, vLLM, Ollama, FastAPI
- Infra: Triton, Docker, HuggingFace Hub

---
## Features

- Scrapes and parses **10-K filings** directly from the SEC EDGAR API
- Cleans and chunks documents into semantically meaningful sections (e.g. *Item 1A. Risk Factors*)
- Builds a **FAISS** vector index using local embeddings (`bge-small-en-v1.5`)
- Serves a **Streamlit UI** powered by a local LLM (Ollama + Llama3/Mistral)
- Cites sources for every answer
- Optional reranker (`bge-reranker-base`) for higher precision retrieval
- Optional RAG evaluation pipeline using **RAGAS**

---

## Tech Stack

| Component | Choice | Notes |
|------------|---------|-------|
| LLM | Ollama (`llama3.1:8b-instruct`) | runs locally on 8GB+ GPU |
| Embeddings | `BAAI/bge-small-en-v1.5` | compact, high-quality |
| Vector DB | FAISS | local, simple |
| Orchestration | LangChain | retrieval + prompt chaining |
| UI | Streamlit | interactive demo |
| Data | SEC 10-K filings | via public EDGAR API |

```mermaid
graph LR
A[User Question] --> B[Retriever (FAISS + Embeddings)]
B --> C[Reranker (optional)]
C --> D[LLM (Ollama / GPT-4o)]
D --> E[Answer + Citations]
```

---

# Example Questions

“What are Apple’s main risk factors in 2023?”

“Summarize Microsoft’s discussion on AI investments.”

“Compare Amazon’s business strategy in the last two 10-K filings.”

---

# Why this stack?
It runs entirely on a laptop, is interview-friendly, and we can swap pieces later.

Orchestration: LangChain or LlamaIndex (pick one; I’ll use LangChain in code below).

Vector store: FAISS (simple, local) → upgrade to Chroma/Weaviate later.

Embeddings (local):

bge-small-en-v1.5 (≈110M) via sentence-transformers — fast + good.

(Optional) gte-base-en for slightly better quality if you have headroom.

Reranker (optional but high impact): bge-reranker-base (cross-encoder) to re-rank top-k passages.

LLM (local): Ollama with:

llama3.1:8b-instruct or mistral:7b (quantized). Your 3080 should handle 7–8B 4-bit quant fine.

LLM (API, if you prefer): GPT-4o-mini / Claude Haiku / Mistral-Large — cheap + strong grounding.

Web/UI: Streamlit (fast to ship) or FastAPI (if you want a clean API).

Eval/observability: ragas + langfuse (optional) for traces.

---

# user setup
```bash
curl -sSL https://install.python-poetry.org | python3 -
# poetry will give you instructions for adding Poetry's directory to your `PATH` environment variable

poetry config virtualenvs.in-project true

poetry install

# in vscode, set python interpreter path to: /path/to/repo/.venv/bin/python

# to install Ollama with sudo rights
# curl -fsSL https://ollama.com/install.sh | sh
# to download prebuilt Ollama binary (if you don't have sudo rights)
# find the appropriate binary from https://github.com/jmorganca/ollama/releases
curl -LO "https://github.com/ollama/ollama/releases/download/v0.12.6/ollama-linux-amd64.tgz"
mkdir ollama/
tar -C ollama/ -xvzf ollama-linux-amd64.tgz
# to serve:
./ollama/bin/ollama serve&
# verify with
./ollama/bin/ollama -v

# run a model
./ollama/bin/ollama run deepseek-r1:latest

source .venv/bin/activate

python3 the_script.py
```


## one time setup when this library was first created
```bash
# after doing the above
poetry init -n
poetry env use $(which python3)
# manually edit python requirement to `requires-python = ">=3.12,<3.15"`
# have to manually fill in dependencies list, description, license etc
# paste black, ruff & pytest config

# add key dependencies
poetry add langchain sentence-transformers streamlit pydantic@^2 tiktoken unstructured[all-docs] pypdf flagembedding ragas datasets fastapi uvicorn httpx tenacity

# add dev dependencies
poetry add -G dev pytest ruff black mypy jupyter
```

# to use faiss-gpu
```bash
# new env with strict channel priority
mamba create -n finrag python=3.11 -y
mamba activate finrag
conda config --env --set channel_priority strict
conda config --env --add channels pytorch
conda config --env --add channels nvidia
conda config --env --add channels conda-forge

# install PyTorch with CUDA 12.1 first (matches FAISS builds well)
mamba install pytorch=2.4.* pytorch-cuda=12.1 -c pytorch -c nvidia -y

# now install a compatible FAISS-GPU build
mamba install faiss-gpu=1.7.4 -c pytorch -y

# sanity check
python - <<'PY'
import faiss, torch
print("FAISS:", faiss.get_version())
print("CUDA available:", torch.cuda.is_available())
PY
```

# conda set up (cannot install faiss-gpu)
```bash
# ensure you already have installed conda such as via miniconda
# change ~/.conda/rag to folder of your choice
conda create --prefix=~/.conda/rag python=3.12
conda activate ~/.conda/rag
```
