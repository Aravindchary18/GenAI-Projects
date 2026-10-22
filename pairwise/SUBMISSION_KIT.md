# Pairwise — submission kit

## Project description (under 200 characters)
Pairwise lets profile-bound AI agents explore adult participants' self-reported compatibility, hold runtime conversations, and rank matches with transparent evidence, uncertainty, and consent-aware controls.

## What to demonstrate
1. **25 participant profiles:** show the synthetic demo label and explain that all 25 are fictional demo personas. Do not call them real people.
2. **Profile analysis:** open one persona and show the fields, evidence trail, and missing-information warnings.
3. **Agent interaction:** select two profiles and run a conversation. Show each agent's turn and execution mode. Local mode is deterministic profile-conditioned generation; optional configured OpenAI mode is LLM-generated.
4. **Rankings:** calculate all 24 alternatives for one selected profile and open the evidence/uncertainty details.
5. **Privacy:** show the separate real dataset workflow and explain that it is disabled by default. No social profiles are scraped.

## 3-minute narration draft

**0:00–0:15 — Hook**  
“What if each dating profile had an AI agent that could explore compatibility before the people ever matched? This is Pairwise, a working agentic matchmaking prototype.”

**0:15–0:35 — Dataset honesty**  
“The demo contains 25 synthetic adult personas, so the website is immediately testable without exposing anyone's private dating data. They are clearly labelled as synthetic—not fabricated real people.”

**0:35–0:55 — Profile analysis**  
“Each profile is structured around self-reported interests, hobbies, values, relationship goals, and lifestyle. Analysis shows its evidence and uncertainty instead of pretending to know things the participant never shared.”

**0:55–1:35 — Agent room**  
“I'll run two separate profile-bound agents. Each turn is generated at runtime from the two profiles and conversation state. The agents surface shared ground and ask questions; they do not make romantic decisions or claim consent for either person.”

**1:35–2:10 — Rankings**  
“Pairwise evaluates every other eligible profile and explains its weighted overlap score, shared fields, possible explicit preference conflicts, and limitations. The score is a transparent demo heuristic, not a scientifically validated prediction of relationship success.”

**2:10–2:35 — Persistence and controls**  
“I can create and delete a profile, and the SQLite-backed demo persists across refreshes. Synthetic and real datasets are kept separate. Real-data mode is disabled by default and requires explicit operator configuration.”

**2:35–3:00 — Responsible data handling**  
“This build does not scrape LinkedIn or Instagram. URLs are participant-provided references only; actual participants must opt in and approve their data. A public real-person launch still needs authentication, participant-level access controls, consent revocation, moderation, and secure managed storage.”

## Technical section: data collection and scraping

**No scraping technology is used in this build.** LinkedIn and Instagram URLs are optional references supplied by the participant/operator. Pairwise does not crawl pages, log into accounts, use unofficial scrapers, or infer profile attributes from third-party pages. The demo is populated by 25 synthetic personas. Real profiles must be contributed or approved by consenting adults; URL presence is not identity verification or consent verification.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn app.api:app --reload --port 8000
# In a second terminal:
streamlit run app/ui.py
```

## Publish a testable demo

1. Create a public GitHub repository and upload the contents of this folder. Do not upload `.env`, API keys, or any real participant data.
2. In Streamlit Community Cloud, create an app from that repository and select `app/ui.py` as the entry point.
3. Keep `ENABLE_REAL_DATA=false` in deployment settings. The synthetic demo works without an LLM key.
4. Wait for the app to build, open the public URL in a private/incognito window, and test all five flows: participants, profile analysis, agent room, rankings, privacy.
5. Copy the actual deployed URL into the submission form. Never submit a guessed URL; a live deployment must be verified after hosting.
6. Record the three-minute walkthrough against the deployed app. Do not claim the synthetic dataset contains real people.

## Important release limitation

This is a runnable challenge prototype, not a production dating service. Public use with real participants requires authentication/authorization, per-person data access controls, consent records and revocation, moderation/reporting, rate limits, encryption and secrets management, backups, legal/privacy review, and a managed database. `ENABLE_REAL_DATA=true` is only an opt-in switch and does not provide these safeguards.
