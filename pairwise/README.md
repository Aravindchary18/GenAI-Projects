# Pairwise — Agentic Dating Website

A consent-first demo of a multi-agent dating workflow. Each participant has a separate profile-bound agent, agents generate messages at runtime, pairwise compatibility is explainable, rankings are persisted, and synthetic demo profiles are isolated from real participant records.

## What is implemented

- 25 **synthetic** adults seeded on first launch; explicitly labelled synthetic.
- Separate `synthetic` and `real` SQLite datasets. Cross-dataset agent runs are rejected.
- Participant creation, adult-only validation, optional LinkedIn/Instagram URL references, real-dataset consent checkbox, and deletion. Real-data mode is disabled by default and requires an explicit operator environment switch.
- Profile analysis from participant-entered/approved form fields with evidence and uncertainty.
- Two profile-bound agents that independently generate each turn from the current profiles and prior conversation. The local mode is deterministic and works without API keys; optional OpenAI mode is available by environment variable.
- Complete rankings against every other participant in the selected dataset; scores show shared fields, conflicts, uncertainties, and method.
- SQLite persistence across app refresh/restart.
- FastAPI endpoints, Streamlit UI, tests, Dockerfile, environment template, and deployment instructions.

## Important truth-in-advertising boundary

This is a runnable prototype, **not a production-ready service**. It does not scrape LinkedIn or Instagram, verify identity, or provide 25 real participants. Public profile URLs are references only. The real dataset workflow requires participant-approved content and an operator attestation of consent; independent consent verification is not implemented. The no-key demo's agents are runtime-generated, profile-conditioned rule-based agents, not an LLM. To use LLM-generated turns, configure the optional OpenAI provider. Compatibility scores are transparent heuristics, not scientifically validated.

The API and UI default to synthetic-only operation. `ENABLE_REAL_DATA=true` deliberately enables the real dataset workflow, but is not itself security: do not set it on a public deployment until authentication and authorization, per-participant access controls, consent receipts and revocation, abuse reporting/moderation, rate limiting, secrets management, encryption at rest/in transit, backups, privacy/legal review, and a managed database are in place. The included consent checkbox is an operator attestation, not a consent-management system.

## Quick start

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Terminal 1 — API:
```bash
uvicorn app.api:app --reload --port 8000
```

Terminal 2 — website:
```bash
streamlit run app/ui.py
```

Open the local Streamlit URL shown in the terminal. The API docs are at `http://127.0.0.1:8000/docs`.

The app seeds 25 synthetic profiles on first use. Use **Synthetic demo** for the instant walkthrough. The database is `data/dating.sqlite3`; change it with `DATING_DB_PATH`.

## Optional LLM mode

Copy `.env.example` to `.env` and export the variables in your shell (Streamlit/Uvicorn do not automatically load `.env` unless you add python-dotenv). Set:
```bash
OPENAI_API_KEY=your_key
USE_LLM_AGENTS=true
USE_LLM_FOR_ANALYSIS=true
OPENAI_MODEL=gpt-4o-mini
```
Install requirements and restart both processes. The agent room displays each turn's execution mode. LLM calls can fail; the application falls back to local profile-conditioned generation. Do not place secrets in the repository.

## Real participant workflow (25+)

1. Recruit 25+ adults directly and explain the purpose, data fields, retention, model provider, and deletion process.
2. Obtain explicit, documented opt-in from every participant before collecting data. The UI checkbox is only an operator attestation, not a consent-management system.
3. Ask each participant to supply or approve their own profile text and public LinkedIn/Instagram URLs. URLs are not proof of identity or consent.
4. Switch the UI dataset to **Consenting adults** and add each adult separately. Use only approved information; do not scrape or bulk-import social profiles.
5. Review every analysis with the participant and correct/remove inaccurate claims.
6. Run agent interactions and rankings only within that dataset. Participants must remain free to opt out; delete their records on request.
7. Do not publish or deploy the real dataset publicly until authentication, authorization, secure storage, consent revocation, and privacy/legal review are complete.

No genuine consenting participant data is bundled. It would be dishonest to fabricate it.

## API overview

- `GET /health`
- `GET /participants?dataset=synthetic|real`
- `POST /participants`
- `GET /participants/{id}`
- `POST /participants/{id}/analyze`
- `DELETE /participants/{id}`
- `POST /agents/run` — run two agents, store turns and evaluation
- `GET /agents/runs?dataset=synthetic&participant_id=...`
- `POST /rankings/{id}` — evaluate every other participant
- `GET /rankings/{id}`
- `POST /demo/reset` — resets only synthetic participants

FastAPI docs: `/docs`.

## Tests

```bash
pytest -q
```

Tests use a temporary SQLite database. The tests cover seeding, runtime agent turns, all-pair ranking, persistence, deletion, invalid links, consent validation, and cross-dataset isolation.

## Deployment guide

### Streamlit Community Cloud (UI)
- Push this repository to a public GitHub repository only with synthetic demo data and no secrets.
- Create a Streamlit app pointing to `app/ui.py`; set Python version to 3.11.
- The default local API URL is for local use. The UI currently calls the shared SQLite/agent modules directly, so it can run without the API process. Keep `ENABLE_REAL_DATA=false` for a public synthetic demo.
- For persistent data, use a persistent database; ephemeral deployment filesystems may reset on rebuild/restart.

### API hosting
- Deploy `uvicorn app.api:app --host 0.0.0.0 --port $PORT` to a Python web service.
- Set environment variables through the host's secret manager.
- A hosted Streamlit UI and API need network configuration and CORS restrictions adjusted to the UI domain.
- SQLite is suitable for a single-instance prototype, not multi-instance production. Move to PostgreSQL and add migrations before production.

### Docker
```bash
docker build -t pairwise .
docker run --rm -p 8000:8000 -p 8501:8501 -v "$PWD/data:/app/data" pairwise
```

The container starts both the API and UI using a lightweight process manager script. Production deployments should separate API and UI services.

## Architecture

- **UI:** Streamlit pages in `app/ui.py`.
- **API:** FastAPI routes in `app/api.py`.
- **Persistence:** SQLite participants, agent runs, rankings; foreign keys cascade on deletion.
- **Agents:** `app/agents.py`; each call uses each participant's own profile, the counterpart's profile, turn number and prior turns. Optional OpenAI provider is environment-controlled.
- **Scoring:** weighted Jaccard overlap across explicit interests, hobbies, values, relationship goals, and lifestyle; participant-entered must-have preferences produce explicit conflict notes.
- **Data access:** form-submitted or participant-approved fields only; social URLs stored as references. No LinkedIn/Instagram scraping or third-party API access.

## Three-minute demo-video storyboard

- **0:00–0:15 — Hook:** “What if every dating profile had its own AI agent?”
- **0:15–0:35 — Dataset honesty:** show “Synthetic demo · 25 profiles”; state they are generated demo personas, not real people.
- **0:35–0:55 — Profile:** open Maya; run profile analysis; show evidence and uncertainty.
- **0:55–1:35 — Live agent room:** choose Maya and Rohan; click Run agents; narrate each runtime-generated turn and identify the local agent mode. If using LLM mode, show the configured LLM mode and disclose provider.
- **1:35–2:10 — Rankings:** rank matches for Maya; inspect score, shared interests/values, incompatibilities, and uncertainty.
- **2:10–2:30 — CRUD and persistence:** create a synthetic participant, refresh, verify it remains, delete it, verify it disappears.
- **2:30–2:45 — Privacy:** show separate real dataset, consent requirement, source links as references only, and deletion.
- **2:45–3:00 — Limitations:** explain no scraping, no real participants bundled, heuristic score not validated, and production security work still needed.

## 200-character project description

Pairwise is a consent-first dating demo where profile-bound AI agents explore compatibility, generate live conversations, and rank matches with explainable evidence, uncertainty, and privacy controls.
