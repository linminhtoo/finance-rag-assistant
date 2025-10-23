from langchain_community.chains import PebbloRetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# TODO: improve prompt to return confidence level, buy/sell recommendation etc
# TODO: improve to agent -> allow web search etc
# reference: https://smith.langchain.com/hub/rlm/rag-prompt
BASE_PROMPT = """You are a top-tier financial analyst working at a world-class hedge fund.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Cite sources as [Title or Filename, page].

Question: {question}
Context:
{context}
Answer:"""

EMB_MODEL = "BAAI/bge-small-en-v1.5"
CHAT_MODEL = "deepseek-r1:latest"
INDEX_PATH = "scripts/artifacts/faiss_index"


def load_vectorstore() -> FAISS:
    emb = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # BGE prefers normalized vectors
    )
    vs = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
    # TODO: how about on-disk index for larger scale?
    return vs


def build_retriever(vs: FAISS, use_reranker: bool = False) -> FAISS:
    base_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    if not use_reranker:
        return base_retriever

    raise NotImplementedError("Reranker integration is currently disabled.")
    # fr_client = Ranker()
    # reranker = FlashrankRerank(client=fr_client, top_n=6)

    # return RerankRetriever(
    #     base_retriever=base_retriever,
    #     rerank_transformer=reranker,
    # )


def make_chain() -> PebbloRetrievalQA:
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)
    vs = load_vectorstore()
    retriever = build_retriever(vs, use_reranker=False)
    prompt = ChatPromptTemplate.from_template(BASE_PROMPT)
    # PROBLEM: PebbloRetrievalQA not compatible with FAISS
    # only works with Pinecone, PineconeVectorStore, PGVector, Qdrant
    return PebbloRetrievalQA.from_chain_type(
        llm=llm,
        app_name="financial-rag-assistant",
        description="A financial RAG assistant that answers questions based on SEC filings and other financial documents.",
        owner="mlin",
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


"""
# https://python.langchain.com/docs/integrations/chat/ollama/#reasoning-models-and-custom-message-roles
# reasoning
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="granite3.2:8b")

messages = [
    ChatMessage(role="control", content="thinking"),
    HumanMessage("What is 3^3?"),
]

response = llm.invoke(messages)
print(response.content)
"""
