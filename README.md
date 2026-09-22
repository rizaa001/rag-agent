# Job Market Research Assistant — Agentic RAG

A single-agent Retrieval-Augmented Generation (RAG) system that helps with job search by deciding, per question, whether to search a local knowledge base of saved job postings or pull live information from the web.

## Why Agentic RAG?

Traditional RAG always retrieves-then-generates, regardless of the question. This project adds a decision layer: the agent evaluates each query and picks the right tool — retrieving from a static, curated dataset when the answer lives there, or reaching out to the live web when it doesn't. This mirrors how a real research assistant would work, rather than blindly searching one fixed source every time.

## Architecture

User Question
│
▼
LLM Agent (Groq — openai/gpt-oss-120b)
│
├── search_postings → ChromaDB vector search over 24 saved job postings
│ (local HuggingFace embeddings, free, no API cost)
│
└── web_search → Tavily live web search
(for current company news, hiring status, market trends)
│
▼
Grounded, tool-sourced answer


## Tech Stack (100% free tier)

| Component | Tool |
|---|---|
| LLM | Groq API — `openai/gpt-oss-120b` |
| Embeddings | HuggingFace `BAAI/bge-base-en-v1.5` (runs locally) |
| Vector DB | ChromaDB (local, persistent) |
| Web Search | Tavily API |
| Orchestration | LangChain (tool-calling agent) |
| UI | Streamlit |

## Evaluation

Built a 12-question evaluation set to test tool-routing accuracy, covering both questions answerable from saved postings and questions requiring live web data.

**Result: 11/12 (92%) correct tool selection.**

The single miss was a defensible edge case — a question phrased ambiguously enough that either tool was a reasonable choice. Full eval script and results are in `run_eval.py` / `eval_questions.py`.

## Features

- Retrieval-augmented answers grounded in real, saved job postings
- Live web search fallback for anything outside the static dataset
- Transparent tool attribution — every answer shows which source it used
- Chat-style Streamlit interface

## Setup

1. Clone this repo and create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install langchain==0.3.27 langchain-community==0.3.27 langchain-core==0.3.72 langchain-groq==0.2.5 langchain-huggingface==0.1.2 langchain-text-splitters==0.3.9 chromadb==0.5.23 sentence-transformers==3.3.1 tavily-python==0.5.0 python-dotenv==1.0.1 streamlit
```

3. Add your free API keys to a `.env` file:

GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here


4. Add job postings as `.txt` files to `docs/`, then build the vector store:
```bash
python ingest.py
```

5. Run the app:
```bash
streamlit run app.py
```

## Project Structure

├── docs/ # Job posting .txt files (knowledge base source)
├── ingest.py # Chunks + embeds postings into ChromaDB
├── tools.py # search_postings and web_search tool definitions
├── agent.py # Tool-calling agent setup
├── eval_questions.py # 12-question evaluation set
├── run_eval.py # Runs eval set, scores tool-routing accuracy
└── app.py # Streamlit chat interface


## Future Improvements

- Add a reranker (cross-encoder) on top of vector search for higher retrieval precision
- Add a claim-verification step to catch unsupported statements before returning an answer
- Expand the knowledge base and add metadata filtering (by role type, seniority, etc.)
