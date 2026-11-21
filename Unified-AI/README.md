# Unified AI Workspace

**An LLM-driven personal AI assistant powered by LangGraph and MCP.**

Unified AI Workspace brings document intelligence, web research, persistent memory, browser automation, and Gmail operations into one assistant. Users interact through natural language, while the LLM determines which tools are needed and LangGraph manages their execution.

The project focuses on building an extensible agent architecture that separates reasoning, workflow orchestration, tool interoperability, and service execution.

---

## Overview

Traditional AI applications often implement individual capabilities as separate workflows. Unified AI Workspace takes a different approach: a single assistant can understand a request, select the appropriate tools, execute multiple operations, inspect their results, and formulate a response.

For example, a user can ask the assistant to research a topic, inspect uploaded documents, remember information, open websites, or prepare an email without manually selecting individual tools.

The system uses:

- **LangGraph** for stateful agent orchestration.
- **LLM tool calling** for dynamic tool selection and decision-making.
- **MCP (Model Context Protocol)** for standardized tool integration.
- **FastAPI** for backend APIs.
- **SQLite** for persistent application data and graph checkpointing.

## Key Capabilities

### 1. Document Intelligence
- PDF document processing and question answering.
- Retrieval-Augmented Generation (RAG).
- Multi-document retrieval capabilities.
- Dense vector retrieval using Qdrant.
- BM25 keyword retrieval.
- Hybrid retrieval and reranking.

### 2. Web Research
- Fresh web research through Tavily.
- Retrieval of external information.
- Research results returned to the assistant for further reasoning.

### 3. Persistent Personal Memory
- Save user information.
- Retrieve stored memories.
- Update existing memories.
- Delete memories.
- SQLite-backed persistence.

### 4. Browser Automation
- Open public websites.
- Play YouTube videos.
- Download public PDF files.
- Fill public form fields.

Form submission is not performed automatically.

### 5. Gmail Integration
- Create Gmail drafts.
- Send drafts through an approval-controlled workflow.
- Validate approval against the exact pending tool call and draft.

Email sending is not permitted without the required approval.

### 6. Career Assistance
- Resume analysis.
- Skill-gap identification.
- Career roadmap generation.

These capabilities reuse the existing backend services.

---

## Architecture

The system follows an LLM-driven agent architecture.

```text
                   USER
                     |
                     v
            Natural Language Request
                     |
                     v
               ASSISTANT LLM
                     |
              Tool Selection
                     |
                     v
              LANGGRAPH AGENT
                     |
              Workflow Routing
                     |
                     v
              MCP TOOL ADAPTER
                     |
                     v
                 MCP CLIENT
                     |
                     v
                 MCP SERVER
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Document    Research   Memory
       Services    Services   Services
          |
          +----------+----------+
                     |
          +----------+----------+
          |                     |
          v                     v
     Browser Tools         Gmail Tools
          |                     |
          |               Approval Gate
          |                     |
          |                     v
          |                 Gmail Send
          |
          v
       Tool Results
          |
          v
       LANGGRAPH
          |
          v
       ASSISTANT LLM
          |
          v
       FINAL RESPONSE
```

### How It Works

1. The user submits a natural-language request.
2. The assistant LLM interprets the request and selects an appropriate tool.
3. LangGraph manages the execution workflow and shared state.
4. The MCP adapter exposes and validates tool calls.
5. The MCP client communicates with the MCP server.
6. The MCP server invokes the corresponding existing backend service.
7. The tool result returns to the assistant.
8. The LLM evaluates the result and decides whether another tool call is necessary.
9. The assistant generates the final response.

This supports sequential tool execution rather than relying on a fixed, predefined workflow for every request.

---

## LangGraph Orchestration

LangGraph manages the assistant's execution lifecycle.

Its responsibilities include:

- Maintaining shared assistant state.
- Invoking the assistant LLM.
- Routing between LLM and tool execution nodes.
- Supporting sequential tool calls.
- Returning tool results to the LLM.
- Handling execution errors.
- Managing Gmail approval workflows.
- Persisting graph checkpoints.

The implementation supports a bounded execution loop:

| Configuration | Limit |
|---|---:|
| Maximum LLM turns | 12 |
| Maximum tool calls | 16 |
| Tool calls per assistant turn | 1 |

The graph also retains a legacy specialist workflow for structured automation requests.

### Important Graph Files

| File | Responsibility |
|---|---|
| `agents/graph.py` | Graph construction, routing, invocation, resumption, checkpointing |
| `agents/llm_nodes.py` | LLM execution, tool selection, routing, tool execution |
| `agents/state.py` | Shared assistant state |
| `agents/nodes.py` | Workflow nodes |
| `agents/specialists.py` | Specialist workflows |
| `agents/tool_registry.py` | Tool registry |
| `agents/mcp_tool_adapter.py` | LLM-facing MCP tool interface |
| `agents/mcp_client.py` | MCP client implementation |

---

## MCP Integration

The project uses the official MCP Python SDK v2.

MCP separates the assistant's tool-selection logic from the actual implementation of individual tools.

### MCP Components

**MCP Server — `backend/mcp_server.py`**

Exposes the assistant's available tools and connects them to existing backend services.

**MCP Client — `backend/agents/mcp_client.py`**

Handles MCP communication, tool discovery, input schemas, and tool execution.

**MCP Tool Adapter — `backend/agents/mcp_tool_adapter.py`**

Connects the LLM-facing tool interface to MCP execution.

### Exposed Tool Interfaces

| Tool | Purpose |
|---|---|
| `list_uploaded_documents` | List uploaded documents |
| `search_uploaded_documents` | Search uploaded documents |
| `research_web_fresh` | Perform fresh web research |
| `remember_user_fact` | Store a user fact |
| `search_user_memory` | Retrieve stored memories |
| `update_user_memory` | Update a stored memory |
| `forget_user_memory` | Delete a stored memory |
| `open_public_website` | Open a public website |
| `play_youtube_video` | Play a YouTube video |
| `download_public_pdf` | Download a public PDF |
| `fill_public_form_fields` | Fill public form fields |
| `create_gmail_draft` | Create a Gmail draft |
| `send_gmail_draft_after_approval` | Send an approved Gmail draft |

The MCP server is designed to reuse existing project services instead of duplicating their implementations.

---

## Human-in-the-Loop Gmail Approval

Email sending is treated as a sensitive operation.

The assistant cannot simply select the Gmail sending tool and execute it immediately.

The approval workflow is:

1. The LLM requests a Gmail send operation.
2. LangGraph detects the sensitive tool call.
3. The graph routes execution to the approval workflow.
4. An approval record is prepared.
5. The user reviews and approves or rejects the operation.
6. The graph resumes after approval.
7. The execution layer validates the approval against the exact tool call and draft.
8. Only a matching approved operation can proceed.

The system also prevents:
- Sending without approval.
- Executing rejected requests.
- Reusing approval for a different draft.
- Executing a tool call with mismatched approval information.

---

## Persistence

SQLite is used for application persistence.

The project includes persistence for:

- Assistant-related data.
- Personal memory.
- Approval records.
- LangGraph execution checkpoints.

The LangGraph SQLite checkpointer supports saving and restoring graph execution state.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Agent orchestration | LangGraph |
| LLM integration | LangChain |
| Tool interoperability | MCP Python SDK v2 |
| Vector database | Qdrant |
| Keyword retrieval | BM25 |
| Embeddings | Sentence Transformers |
| Reranking | BAAI/bge-reranker-base |
| Web research | Tavily |
| Browser automation | Playwright |
| Persistence | SQLite |
| Email integration | Gmail API |
| Graph checkpointing | LangGraph SQLite Checkpointer |
| Deployment | Docker / Docker Compose |

---

## Project Structure

```text
Unified-AI/
│
├── backend/
│   │
│   ├── agents/
│   │   ├── graph.py
│   │   ├── llm_nodes.py
│   │   ├── mcp_client.py
│   │   ├── mcp_tool_adapter.py
│   │   ├── nodes.py
│   │   ├── specialists.py
│   │   ├── state.py
│   │   └── tool_registry.py
│   │
│   ├── services/
│   │   ├── agent_service.py
│   │   ├── approval_service.py
│   │   ├── assistant_llm.py
│   │   ├── assistant_store.py
│   │   ├── bm25_service.py
│   │   ├── cache_service.py
│   │   ├── embedding_service.py
│   │   ├── fresh_research_service.py
│   │   ├── gmail_service.py
│   │   ├── hybrid_retrieval_service.py
│   │   ├── memory_service.py
│   │   ├── rag_service.py
│   │   ├── reranker_service.py
│   │   ├── retrieval_service.py
│   │   ├── roadmap_service.py
│   │   ├── skill_gap_service.py
│   │   └── streaming_service.py
│   │
│   ├── tools/
│   │   └── browser_tools.py
│   │
│   ├── evals/
│   │   ├── results/
│   │   ├── tests/
│   │   ├── agent_evaluator.py
│   │   ├── agent_tasks.jsonl
│   │   ├── rag_evaluation.py
│   │   └── resume_evaluation.py
│   │
│   ├── mcp_server.py
│   └── tests/
│
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11
- Git
- Access to the required external API services
- Qdrant configuration
- Gmail API credentials if using Gmail functionality

### 1. Clone the Repository

```bash
git clone https://github.com/Aravindchary18/GenAI-Projects.git
cd GenAI-Projects/Unified-AI/backend
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Use the project's requirements file:

```powershell
pip install -r requirements.txt
```

Ensure the environment includes the required agent dependencies:

```powershell
pip install "mcp>=2,<3" langgraph langgraph-checkpoint-sqlite
```

### 4. Configure Environment Variables

Create the appropriate `.env` file using the variable names expected by the existing application.

Configure the credentials required for the services you intend to use, including:

- LLM provider.
- Tavily.
- Gmail API and OAuth.
- Qdrant, if configured through environment variables.

Never commit API keys, OAuth credentials, or other secrets.

---

## Running the Application

The application uses FastAPI for backend APIs and Streamlit for the user interface.

Start each component using the existing application entry points and configuration.

The MCP server must also be available to the MCP client using the configured transport.

Exact launch commands depend on the project's current entry-point and configuration files.

---

## Testing

The project includes unit tests for agent orchestration and evaluation.

### Agent Graph Tests

Run from the `backend` directory:

```powershell
python -m unittest discover -s .\evals\tests -p "test_agent_graph.py" -v
```

**Latest verified result: 8/8 tests passed.**

The tests cover:

- LLM tool selection and final response.
- Multiple sequential tool calls.
- Tool execution failures.
- Gmail approval routing.
- Blocking unapproved sending.
- Preventing rejected sending.
- Matching approvals to the correct draft.

### Agent Evaluator Tests

```powershell
python -m unittest discover -s evals/tests -p "test_agent_evaluator.py" -v
```

**Latest verified result: 3/3 tests passed.**

The evaluator checks recorded execution results against expected tools, routing, completion, approval requirements, citations, and latency.

### Current Test Results

| Test Suite | Passed |
|---|---:|
| Agent Graph | 8/8 |
| Agent Evaluator | 3/3 |
| **Total** | **11/11** |

These results represent the verified test suites, not a claim that every external integration has passed end-to-end testing.

---

## Evaluation

The agent evaluation dataset is located at:

```text
backend/evals/agent_tasks.jsonl
```

It currently contains six task categories:

- Document search.
- Fresh web research.
- Website inspection.
- Gmail draft creation.
- Form filling.
- Approval-controlled Gmail sending.

The evaluator reads execution results rather than independently running live tools.

---

## Design Principles

The project follows several engineering principles:

1. **Separation of concerns:** LLM reasoning, graph orchestration, MCP communication, and service execution remain separate.
2. **Reusable services:** Existing backend functionality is reused instead of duplicated.
3. **Dynamic tool selection:** The LLM determines which tools are relevant to a request.
4. **Stateful execution:** LangGraph manages execution state and checkpoints.
5. **Controlled side effects:** Sensitive actions require explicit approval.
6. **Bounded execution:** Tool calls and LLM turns have configured limits.
7. **Testability:** Agent routing, error handling, and approval behavior have dedicated tests.

---

## Current Development Status

The following components have been implemented and their relevant unit tests have passed:

- LLM-driven LangGraph orchestration.
- MCP client and tool adapter.
- MCP server implementation.
- Sequential tool execution.
- Gmail approval routing and validation.
- Offline agent evaluation.
- SQLite graph checkpoint integration.

The graph and evaluator test suites currently have **11 passing tests combined**.

Further integration testing is needed to verify every MCP tool against its actual backend service and to validate complete application workflows with real external services.

---

## Future Improvements

- Complete end-to-end testing of all MCP tools.
- Strengthen multi-document retrieval and evaluation.
- Improve tool execution observability.
- Expand agent evaluation datasets.
- Improve failure recovery and retry handling.
- Validate persistent graph resumption across real sessions.
- Complete deployment and production-readiness checks.

---

## Project Goal

The goal of Unified AI Workspace is to build a unified, extensible personal AI assistant where an LLM can understand user intent, coordinate tools through LangGraph, access capabilities through MCP, and execute useful workflows while maintaining control over sensitive actions.
