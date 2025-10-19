"""
Quality upgrades (pick 1-2 if time permits)

Query Rewriting (HyDE / Multi-query): expand the user query into paraphrases, retrieve for each, merge, rerank.

Metadata filters: allow doc_type:guidance, year:>=2022, etc.

Section-aware chunking: split by headings first; store section_title.

Answer verification: second pass that checks whether each sentence is supported by citations.

-----

If you prefer LlamaIndex (equally good), it's a 10-line swap:
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader etc. 
LlamaIndex also has great RAG eval & observability add-ons.
"""
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_PROMPT = """You are a careful assistant. Answer using only the provided context.
If the answer is not contained in the context, say you don't know.
Cite sources as [Title or Filename, page]. Keep answers concise.

Question: {question}
Context:
{context}
Answer:"""

def load_vectorstore():
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", normalize_embedding=True
    )
    vs = FAISS.load_local("artifacts/faiss_index", emb, allow_dangerous_deserialization=True)
    return vs

def build_retriever(vs, use_reranker=True):
    base = vs.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    if not use_reranker:
        return base
    compressor = CrossEncoderReranker(
        model="BAAI/bge-reranker-base",
        top_n=6,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )

def make_chain():
    llm = ChatOllama(model="llama3.1:8b-instruct", temperature=0.2)
    vs = load_vectorstore()
    retriever = build_retriever(vs, use_reranker=True)
    prompt = ChatPromptTemplate.from_template(BASE_PROMPT)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
