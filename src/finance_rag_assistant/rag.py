from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# v1.0 agent + tools APIs
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model  # universal initializer (v1.0)
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Config
# -----------------------------
EMB_MODEL = "BAAI/bge-small-en-v1.5"
# Use a tool-calling capable model here. If deepseek-r1 doesn't tool-call,
# switch to a tool-capable Ollama model like llama3.1 or qwen2.5.
# CHAT_MODEL = "ollama:deepseek-r1:latest"
# deepseek v3.1 is tool-capable
# reference: https://ollama.com/library/deepseek-v3.1
# this is 404 GB, quite big...
# CHAT_MODEL = "ollama:deepseek-v3.1:latest"
# this is 13 GB, better...
CHAT_MODEL = "ollama:gpt-oss:20b"
INDEX_PATH = "scripts/artifacts/faiss_index"

SYSTEM_PROMPT = """You are a top-tier financial analyst at a world-class hedge fund.
When answering, if the user's query requires domain facts from our knowledge base,
CALL the `kb_search` tool to retrieve passages. Ground answers in those passages.
Cite sources inline as [Filename or Title, page]. If unsure, say you don't know.
"""


# -----------------------------
# Vector store / retriever
# -----------------------------
def load_vectorstore() -> FAISS:
    emb = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # BGE prefers normalized vectors
    )
    # allow_dangerous_deserialization=True because we created the index locally
    return FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)


def build_retriever(vs: FAISS):
    # tune k as desired; agent may call the tool multiple times
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 8})


# -----------------------------
# Retrieval tool (v1.0)
# Return BOTH a readable string AND the raw docs as an artifact so the UI
# can list sources without parsing LLM output.
# -----------------------------
def make_kb_tool(retriever):
    @tool("kb_search", response_format="content_and_artifact")
    def kb_search(query: str) -> Tuple[str, List[dict]]:
        """Search the internal document index for relevant passages to answer finance questions."""
        docs: List[Document] = retriever.invoke(query)
        # Produce readable content the model can quote from
        chunks = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            page_str = f", page {page}" if page is not None else ""
            chunks.append(f"[{i}] {d.page_content}\nSource: {src}{page_str}")
        content = "\n\n".join(chunks) if chunks else "No results."

        # Also return the raw docs as an artifact for the UI
        artifact = [{"page_content": d.page_content, "metadata": dict(d.metadata)} for d in docs]
        return content, artifact

    return kb_search


# -----------------------------
# Streamlit adapter so that our UI can keep calling chain({"query": ...})
# and get {"result": ..., "source_documents": [...]}
# -----------------------------
@dataclass
class _StreamlitAgentAdapter:
    agent: Any
    tool_name: str = "kb_search"

    def __call__(self, inputs: Dict[str, Any]):
        q = inputs.get("query") or inputs.get("input") or ""
        resp = self.agent.invoke({"messages": [HumanMessage(content=q)]})

        # Final answer: last assistant message
        messages = resp.get("messages", [])
        final_text = ""
        for m in reversed(messages):
            # AIMessage-like objects have 'type' == 'ai'
            if getattr(m, "type", None) == "ai":
                final_text = getattr(m, "content", "") or ""
                break

        # Try to find the most recent ToolMessage from our retriever tool
        source_documents: List[Document] = []
        for m in reversed(messages):
            if getattr(m, "type", None) == "tool" and getattr(m, "name", "") == self.tool_name:
                # If our tool used response_format="content_and_artifact",
                # the artifact field holds our raw docs (list of dicts)
                artifact = getattr(m, "artifact", None)
                if isinstance(artifact, list):
                    for a in artifact:
                        md = a.get("metadata", {}) if isinstance(a, dict) else {}
                        pc = a.get("page_content", "") if isinstance(a, dict) else ""
                        source_documents.append(Document(page_content=pc, metadata=md))
                break

        return {"result": final_text, "source_documents": source_documents}


# -----------------------------
# Factory
# -----------------------------
def make_chain():
    vs = load_vectorstore()
    retriever = build_retriever(vs)
    kb_tool = make_kb_tool(retriever)

    # v1.0: create_agent is the standard agent constructor
    # see: https://reference.langchain.com/python/langchain/models/
    # https://docs.langchain.com/oss/python/integrations/chat/ollama
    # https://docs.langchain.com/oss/python/langchain/models#initialize-a-model
    llm = init_chat_model(
        CHAT_MODEL,
        temperature=0.2,
        # first start the server with `./ollama/bin/ollama serve`
        # change the path as needed
        base_url="http://127.0.0.1:11434",
    )

    agent = create_agent(
        model=llm,
        tools=[kb_tool],
        system_prompt=SYSTEM_PROMPT,
        name="Financial RAG Agent",
    )

    return _StreamlitAgentAdapter(agent)
