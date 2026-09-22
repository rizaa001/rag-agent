import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# --- Tool 1: search_postings (your RAG retriever) ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def search_postings_func(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant postings found in the saved data."
    
    results = []
    for doc in docs:
        role = doc.metadata.get("role", "unknown")
        results.append(f"[{role}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)

search_postings = Tool(
    name="search_postings",
    func=search_postings_func,
    description="Use this to answer questions about saved job postings, required skills, qualifications, or interview questions from the local knowledge base. Best for questions like 'what skills are required' or 'what does this role need'."
)

# --- Tool 2: web_search (Tavily) ---
web_search = TavilySearchResults(
    max_results=3,
    name="web_search",
    description="Use this for current company news, hiring status, recent job market trends, or anything not covered in the saved postings. Best for questions about 'is X hiring now' or 'recent news about X'."
)

# --- Test block ---
if __name__ == "__main__":
    result = search_postings_func("what skills are needed for AI automation roles")
    print(result)