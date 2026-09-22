import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "docs"

# Step 1: Load all .txt files and attach metadata from filename
all_docs = []
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_DIR, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()

        # filename like "ai_engineer.txt" -> role="ai engineer"
        role = filename.replace(".txt", "").replace("_", " ").replace("-", " ").strip()

        for doc in docs:
            doc.metadata["role"] = role
            doc.metadata["source_file"] = filename

        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} documents from {DOCS_DIR}/")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks")

# Step 3: Embed locally (free, no API needed) and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Done. Vector store saved to ./chroma_db")