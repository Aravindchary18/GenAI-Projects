# Unified AI — Enterprise Document Intelligence & Career Assistant

An end-to-end **Applied AI platform** combining advanced **Retrieval-Augmented Generation (RAG), hybrid search, cross-encoder reranking, LLM orchestration, and agentic web search**.

Built with FastAPI, LangChain, Qdrant, BM25, Sentence Transformers, and multiple LLM providers, the platform brings document intelligence, career analysis, and web research into one workspace.

## Key Features

- **Advanced RAG:** PDF ingestion, semantic chunking, vector retrieval, and context-grounded generation.
- **Hybrid Retrieval:** Combines Qdrant vector search with BM25 keyword search.
- **Cross-Encoder Reranking:** Uses `BAAI/bge-reranker-base` to improve retrieval relevance.
- **LLM Fallbacks:** BharatCode as the primary provider with local Ollama fallback for RAG generation.
- **Agentic Web Search:** LLM-powered agent with Tavily search tool integration.
- **Career Intelligence:** Resume analysis, job-description skill-gap analysis, and personalized career roadmaps.
- **Semantic Caching:** SQLite-backed cache for semantically similar web-search queries.
- **Streaming Responses:** FastAPI streaming for incremental LLM responses.
- **Containerized Architecture:** Docker Compose integration for backend, frontend, Qdrant, and Ollama.

## Architecture

```text
                   ┌──────────────────────┐
                   │   Streamlit Frontend │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │     FastAPI Backend  │
                   └──────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
  ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
  │  Document RAG  │  │ Career Mentor  │  │  Web Research  │
  └───────┬────────┘  └────────────────┘  └───────┬────────┘
          │                                       │
  ┌───────▼────────┐                     ┌────────▼────────┐
  │ PDF Extraction │                     │  LLM Agent      │
  │ & Chunking     │                     │  + Tavily Tool  │
  └───────┬────────┘                     └────────┬────────┘
          │                                       │
  ┌───────▼────────┐                     ┌────────▼────────┐
  │ MiniLM         │                     │ SQLite Semantic │
  │ Embeddings     │                     │ Cache           │
  └───────┬────────┘                     └─────────────────┘
          │
  ┌───────▼────────┐
  │ Qdrant + BM25  │
  │ Hybrid Search  │
  └───────┬────────┘
          │
  ┌───────▼────────┐
  │ Cross-Encoder  │
  │ Reranking      │
  └───────┬────────┘
          │
  ┌───────▼────────┐
  │ Relevant       │
  │ Context        │
  └───────┬────────┘
          │
  ┌───────▼────────┐
  │ LLM Generation │
  │ + Streaming    │
  └────────────────┘
```

## 1. Advanced RAG Pipeline

The document Q&A system uses a hybrid retrieval pipeline to retrieve relevant information from uploaded PDFs.

### Document Processing

1. Extract text from PDFs using `pdfplumber`.
2. Split extracted text using `RecursiveCharacterTextSplitter`.
3. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Store vectors and document metadata in Qdrant.

| Configuration | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Chunk size | 500 characters |
| Chunk overlap | 100 characters |
| Vector database | Qdrant |
| Keyword retrieval | BM25 |

### Hybrid Retrieval & Reranking

The retrieval pipeline combines two complementary search methods:

- **Dense retrieval:** Qdrant vector similarity search retrieves semantically relevant chunks.
- **Sparse retrieval:** BM25 retrieves keyword-relevant chunks.
- **Reranking:** `BAAI/bge-reranker-base` scores retrieved candidates based on query-document relevance.
- **Context selection:** The top 5 reranked chunks are passed to the generation model.

Document-level filtering uses metadata such as `user_id` and `document_id`.

### LLM Generation

- BharatCode is configured as the primary LLM provider.
- Local Ollama provides a fallback.
- LangChain's fallback mechanism handles provider failures.
- Responses are streamed through FastAPI's `StreamingResponse`.

Production LLM configuration:

```python
llm = bharath_llm.with_fallbacks([local_llm])
```

## 2. Agentic Web Research

A tool-using LLM agent performs web research through Tavily.

### Workflow

1. Receives a user research query.
2. Uses the web-search tool to retrieve external information.
3. Processes search results through the LLM.
4. Generates a response based on retrieved information.
5. Checks the semantic cache to reuse previously generated answers for sufficiently similar queries.

### Semantic Caching

SQLite stores cached queries, embeddings, and responses.

- Embeddings generated using `all-MiniLM-L6-v2`.
- Normalized embeddings are compared using dot-product similarity.
- Similarity threshold: `0.85`.
- Cache matches can avoid redundant LLM inference.

## 3. Career Intelligence

The Career Mentor module provides three AI-powered workflows.

### Resume Analysis

Extracts and evaluates:

- Technical and soft skills
- Experience and education
- Projects and strengths
- Weaknesses and improvement areas
- Resume score and recommendations

### Job Description Skill-Gap Analysis

Compares a resume against a target job description to identify:

- Matching skills
- Missing skills
- Skill gaps
- ATS-oriented feedback
- Recommended improvements and learning areas

### Personalized Career Roadmap

Generates a structured learning roadmap containing:

- Current skill assessment
- Foundation, intermediate, and job-ready phases
- Recommended projects
- Interview preparation
- Learning resources

## 4. Evaluation Results

The project includes lightweight evaluation scripts for RAG and web-agent workflows.

### RAG Evaluation

15 test cases evaluated using expected-keyword coverage.

| LLM Provider | Passed | Pass Rate |
|---|---:|---:|
| BharatCode | 11/15 | 73.33% |
| Ollama — Qwen2.5 Coder 3B | 6/15 | 40.00% |

### Web-Agent Evaluation

12 test cases evaluating tool invocation and expected-keyword coverage.

| Metric | BharatCode | Ollama — Qwen3 4B |
|---|---:|---:|
| Passed tests | 9/12 | 10/12 |
| Pass rate | 75.00% | 83.33% |
| Tool invocation rate | 83.33% | 100% |
| Expected-keyword coverage | 75.00% | 80.56% |
| Average latency | 22.71s | 26.79s |

**Evaluation notes:**
- These are lightweight functional and expected-keyword evaluations, not comprehensive factual-accuracy benchmarks.
- RAG pass rates measure expected-keyword coverage.
- Web-agent tests also evaluate whether the agent invokes its search tool.
- BharatCode web-agent tests `web_004` and `web_005` encountered provider HTTP 503 errors.
- Ollama web-agent tests `web_008` and `web_012` failed.

## 5. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Backend health check |
| POST | `/upload` | Upload and process PDF documents |
| POST | `/chat` | Ask questions about uploaded documents |
| POST | `/websearch/search` | Perform agentic web research |
| POST | `/career_mentor/analyze` | Analyze a resume |
| POST | `/career_mentor/skill-gap` | Compare resume against a job description |
| POST | `/career_mentor/roadmap` | Generate a personalized career roadmap |

## 6. Technology Stack

| Category | Technologies |
|---|---|
| Languages | Python, SQL |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM Integration | LangChain, BharatCode, Ollama |
| Local LLMs | Qwen2.5 Coder 3B, Qwen3 4B |
| Vector Database | Qdrant |
| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 |
| Retrieval | Dense Vector Search, BM25 |
| Reranking | BAAI/bge-reranker-base |
| Web Search | Tavily |
| Caching | SQLite, Semantic Similarity |
| Document Processing | pdfplumber, RecursiveCharacterTextSplitter |
| Deployment | Docker, Docker Compose |

## 7. Project Structure

```text
Unified-AI/
│
├── backend/
│   ├── data/
│   ├── evals/
│   ├── routes/
│   │   └── upload.py
│   │
│   ├── services/
│   │   ├── agent_service.py
│   │   ├── bm25_service.py
│   │   ├── rag_service.py
│   │   ├── retrieval_service.py
│   │   ├── reranker_service.py
│   │   ├── resume_analysis_service.py
│   │   ├── skill_gap_service.py
│   │   ├── roadmap_service.py
│   │   └── streaming_service.py
│   │
│   ├── tools/
│   │   ├── web_search.py
│   │   └── voice.py
│   │
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   └── streamlit_app.py
│
├── docker-compose.yml
└── README.md
```

## 8. Engineering Focus

This project explores practical engineering challenges in building LLM-powered applications:

- Designing hybrid retrieval pipelines instead of relying exclusively on vector similarity.
- Improving retrieved-context relevance through cross-encoder reranking.
- Integrating hosted and local LLM providers with fallback handling.
- Building tool-using agents for external information retrieval.
- Reducing repeated inference through semantic caching.
- Exposing AI workflows through structured REST APIs and streaming responses.
- Evaluating model behavior using repeatable test cases.

---

**Built as a hands-on Applied AI engineering project focused on RAG systems, LLM applications, retrieval pipelines, and agentic workflows.**
