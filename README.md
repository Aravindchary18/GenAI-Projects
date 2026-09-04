# Unified AI — Enterprise Document AI Workspace

A production-focused Generative AI application that combines **Retrieval-Augmented Generation (RAG), hybrid retrieval, cross-encoder reranking, LLM-powered career intelligence, and a practical web-search agent** in one workspace.

The system demonstrates how modern LLM applications can be built beyond a basic chatbot, combining document ingestion, semantic and keyword retrieval, relevance reranking, grounded generation, streaming responses, tool use, semantic caching, REST APIs, and containerized services.

## What It Does

* **Document Q&A** — Upload a text-based PDF and ask questions grounded in its content.
* **Hybrid RAG** — Combines semantic vector retrieval with BM25 keyword retrieval.
* **Cross-Encoder Reranking** — Reranks retrieved chunks to improve relevance before generation.
* **Streaming LLM Responses** — Streams RAG-generated answers from the local LLM to the client.
* **Web Search Agent** — Uses a single LLM agent with a Tavily web-search tool to retrieve external information.
* **Resume Analysis** — Analyzes a resume and returns skills, level, projects, education, experience, strengths, weaknesses, improvements, and a score.
* **Skill Gap Analysis** — Compares a resume against a job description and identifies matched skills, missing skills, ATS score, improvements, and learning recommendations.
* **Career Roadmap** — Generates a structured roadmap based on current skills, target role, and experience.
* **Semantic Response Cache** — Reuses cached answers for sufficiently similar web-search questions to reduce repeated LLM generation.
* **REST API** — FastAPI exposes the application's main capabilities through API endpoints.
* **Containerized Architecture** — Backend, frontend, Qdrant, and Ollama run as separate Docker Compose services.

## Architecture

```text
                         ┌──────────────────────┐
                         │   Streamlit Frontend │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     FastAPI API      │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
       Document RAG            Career Mentor          Web Search
             │                      │                      │
             │                      │                ┌─────▼─────┐
             │                      │                │ LLM Agent │
             │                      │                └─────┬─────┘
             │                      │                      │
             │                      │                Tavily Search
             │                      │
             ▼                      ▼
        PDF ingestion          Ollama / Qwen
             │
             ▼
       Text chunking
             │
             ▼
      ┌───────────────┐
      │  Embeddings   │
      │ MiniLM 384-D  │
      └───────┬───────┘
              │
              ▼
           Qdrant
              │
       ┌──────┴──────┐
       ▼             ▼
 Vector Retrieval   BM25
       │             │
       └──────┬──────┘
              ▼
       Hybrid Retrieval
              │
              ▼
       Cross-Encoder
          Reranking
              │
              ▼
        Relevant Context
              │
              ▼
      Qwen 2.5 Coder 3B
              │
              ▼
       Streaming Answer
```

## RAG Pipeline

The document question-answering pipeline follows:

```text
PDF
 ↓
Text Extraction
 ↓
Recursive Character Chunking
 ↓
Sentence-Transformer Embeddings
 ↓
Qdrant Vector Storage
 ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Vector Search           BM25
 │                       │
 └───────────┬───────────┘
             ▼
      Hybrid Retrieval
             │
             ▼
    Cross-Encoder Reranker
             │
             ▼
      Top Relevant Chunks
             │
             ▼
      Context Construction
             │
             ▼
       Qwen 2.5 Coder
             │
             ▼
      Streaming Response
```

### Retrieval Design

* **PDF extraction:** `pdfplumber`
* **Chunking:** `RecursiveCharacterTextSplitter`
* **Chunk size:** 500 characters
* **Chunk overlap:** 100 characters
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Embedding dimension:** 384
* **Vector database:** Qdrant
* **Keyword retrieval:** BM25
* **Initial retrieval:** top 10 candidates from vector search and top 10 from BM25
* **Reranker:** `BAAI/bge-reranker-base`
* **Reranked context:** top 5 candidates
* **LLM:** Qwen 2.5 Coder 3B Instruct via Ollama

Retrieved chunks are filtered by `document_id`, keeping document Q&A scoped to the selected uploaded document.

## Agent / Web Search

The application also includes a practical **single LLM agent with a web-search tool**.

```text
User Question
     ↓
Semantic Cache Check
     ↓
LLM Agent
     ↓
Web Search Tool
     ↓
Tavily Search
     ↓
Search Results
     ↓
LLM Response
     ↓
Semantic Cache
```

The agent uses `qwen2.5:3b-instruct` through Ollama and is configured with a Tavily web-search tool.

The web-search system prompt instructs the agent to call the search tool before answering and to base its response on the returned search results.

The web-search workflow uses a lightweight semantic cache backed by SQLite.

## Career Intelligence

The backend also provides three LLM-powered career workflows.

### Resume Analysis

```text
Resume PDF
   ↓
PDF Text Extraction
   ↓
Qwen 2.5 Coder 3B
   ↓
Structured JSON
```

Returns:

* Skills
* Level
* Projects
* Education
* Experience
* Strengths
* Weaknesses
* Resume improvements
* Score

### Skill Gap Analysis

```text
Resume + Job Description
          ↓
       Qwen 2.5 Coder 3B
          ↓
    Structured Analysis
```

Produces:

* Matched skills
* Missing skills
* ATS score
* Resume improvements
* Learning recommendations

### Career Roadmap

```text
Current Skills
      +
Target Role
      +
Experience
      ↓
Qwen 2.5 Coder 3B
      ↓
Structured Roadmap
```

The roadmap contains:

* Current skill assessment
* Foundation phase
* Intermediate phase
* Job Ready phase
* Interview preparation
* Portfolio projects
* Learning resources

## Caching

The web-search agent uses a lightweight semantic cache backed by SQLite.

Questions are embedded using the same `all-MiniLM-L6-v2` embedding model with normalized embeddings. Similarity is then calculated using a dot product between the normalized vectors.

A cached response is reused when the similarity reaches the configured threshold of **0.85**.

This can reduce repeated LLM generation for semantically similar web-search questions.

## API Endpoints

| Endpoint                        | Purpose                                    |
| ------------------------------- | ------------------------------------------ |
| `POST /upload`                  | Upload and index a PDF document            |
| `POST /chat`                    | Ask questions against an uploaded document |
| `POST /websearch/search`        | Run the web-search agent                   |
| `POST /career_mentor/analyze`   | Analyze a resume                           |
| `POST /career_mentor/skill-gap` | Compare a resume with a job description    |
| `POST /career_mentor/roadmap`   | Generate a career roadmap                  |
| `GET /`                         | Backend root endpoint                      |

## Tech Stack

### GenAI / LLM

* Qwen 2.5 Coder 3B Instruct
* Qwen 2.5 3B Instruct for the web-search agent
* Ollama
* LangChain
* Agent/tool calling
* Tavily Search

### RAG / Retrieval

* Sentence Transformers
* `all-MiniLM-L6-v2`
* Qdrant
* BM25
* Cross-Encoder Reranking
* Recursive Character Text Splitting

### Backend

* Python
* FastAPI
* Pydantic
* `StreamingResponse`
* REST APIs

### Frontend

* Streamlit

### Infrastructure

* Docker
* Docker Compose
* Qdrant container
* Ollama container
* SQLite semantic cache

## Project Structure

```text
Unified-AI/
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── upload.py
│   │   ├── career_mentor.py
│   │   └── web_search.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py
│   │   ├── bm25_service.py
│   │   ├── cache_service.py
│   │   ├── embedding_service.py
│   │   ├── hybrid_retrieval_service.py
│   │   ├── rag_service.py
│   │   ├── reranker_service.py
│   │   ├── retrieval_service.py
│   │   ├── resume_analysis_service.py
│   │   ├── skill_gap_service.py
│   │   ├── roadmap_service.py
│   │   └── streaming_service.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── web_search.py
│   └── utils/
│       ├── __init__.py
│       ├── chunking.py
│       ├── logging_utils.py
│       └── prompts.py
├── frontend/
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

## Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd GenAI-Projects/Unified-AI
```

### 2. Configure environment variables

Create a `.env` file containing the required Tavily API key:

```env
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Start the services

```bash
docker compose up --build
```

The Docker Compose setup starts:

* FastAPI backend
* Streamlit frontend
* Qdrant
* Ollama

### 4. Access the application

Frontend:

```text
http://localhost:8501
```

Backend:

```text
http://localhost:8000
```

Qdrant:

```text
http://localhost:6333
```

## Engineering Focus

This project focuses on practical LLM application engineering rather than model training.

Key engineering areas demonstrated:

* Retrieval-Augmented Generation
* Hybrid search
* Semantic embeddings
* Vector databases
* Cross-encoder reranking
* LLM integration
* Agent/tool use
* Prompt constraints
* Streaming inference
* Semantic caching
* REST API design
* PDF document processing
* Dockerized services
* Modular backend architecture

## Limitations

* PDF ingestion currently targets text-based PDFs.
* The web-search workflow depends on a Tavily API key.
* LLM inference is performed locally through Ollama and therefore depends on available system resources.
* The current project is a portfolio implementation and is not presented as a production SaaS deployment with enterprise-scale observability, authentication, or distributed infrastructure.

## Why This Project?

The goal was to build an end-to-end **practical GenAI system** that demonstrates how individual LLM application components fit together.

**Document workflow:**

```text
documents
    ↓
retrieval
    ↓
reranking
    ↓
grounded generation
```

**Tool-using workflow:**

```text
LLM agent
    ↓
web-search tool
    ↓
Tavily search
    ↓
response
```

**Application architecture:**

```text
API
 ↓
services
 ↓
AI components
 ↓
containers
 ↓
user-facing application
```

Rather than treating an LLM as a standalone chatbot, the project focuses on integrating retrieval, reranking, generation, tool use, caching, APIs, and infrastructure into a single practical application.

---

```text
Built as a practical Generative AI / Applied AI engineering project.
```
