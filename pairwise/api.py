from fastapi import FastAPI, HTTPException
import os
from fastapi.middleware.cors import CORSMiddleware
from . import db
from .models import ParticipantCreate, AgentRunRequest
from .agents import analyze_profile, run_pair, compatibility
import json

db.init_db()
app=FastAPI(title="Agentic Dating API", version="1.0.0",
 description="Consent-first, self-report-only dating agent demo. Synthetic and real datasets are isolated.")
_cors_origins = [x.strip() for x in os.getenv("DATING_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "Authorization"]) 

# Public deployments should remain synthetic-only unless an operator deliberately opts in.
# This is a safety switch, not a substitute for authentication/authorization.
def _require_real_data_enabled(dataset: str):
    if dataset == "real" and os.getenv("ENABLE_REAL_DATA", "false").lower() != "true":
        raise HTTPException(403, "Real-participant data is disabled on this deployment. Use the synthetic demo or configure ENABLE_REAL_DATA=true only after adding access controls.")

@app.get("/health")
def health(): return {"status":"ok","persistence":"SQLite","real_social_scraping":False}

@app.get("/participants")
def participants(dataset: str="synthetic"):
    if dataset not in ("synthetic","real"): raise HTTPException(400,"dataset must be synthetic or real")
    _require_real_data_enabled(dataset)
    return db.list_participants(dataset)

@app.post("/participants")
def create_participant(p: ParticipantCreate):
    _require_real_data_enabled(p.dataset)
    if p.dataset=="real" and not p.consent_confirmed:
        raise HTTPException(400,"Real participants must confirm consent.")
    if p.dataset=="real" and not (p.linkedin_url or p.instagram_url):
        raise HTTPException(400,"Real participant needs at least one profile URL; URL presence is not identity verification.")
    data=p.model_dump()
    # Source links are stored as user-provided references, never scraped automatically.
    return db.insert_participant(data)

@app.get("/participants/{pid}")
def participant(pid: str):
    p=db.get_participant(pid)
    if not p: raise HTTPException(404,"Participant not found")
    _require_real_data_enabled(p["dataset"])
    return p

@app.post("/participants/{pid}/analyze")
def analyze(pid: str):
    p=db.get_participant(pid)
    if not p: raise HTTPException(404,"Participant not found")
    _require_real_data_enabled(p["dataset"])
    result=analyze_profile(p); db.save_analysis(pid,result)
    return result

@app.delete("/participants/{pid}")
def delete(pid: str):
    p=db.get_participant(pid)
    if not p: raise HTTPException(404,"Participant not found")
    _require_real_data_enabled(p["dataset"])
    if not db.delete_participant(pid): raise HTTPException(404,"Participant not found")
    return {"deleted":True,"participant_id":pid}

@app.post("/agents/run")
def run_agents(req: AgentRunRequest):
    a=db.get_participant(req.participant_a_id); b=db.get_participant(req.participant_b_id)
    if not a or not b: raise HTTPException(404,"Both participants must exist")
    _require_real_data_enabled(a["dataset"]); _require_real_data_enabled(b["dataset"])
    if a["id"]==b["id"]: raise HTTPException(400,"Choose two different participants")
    if a["dataset"]!=b["dataset"]: raise HTTPException(400,"Synthetic and real participants cannot be mixed")
    if a["age"]<18 or b["age"]<18: raise HTTPException(400,"All participants must be adults")
    turns,evaluation=run_pair(a,b,req.max_turns)
    rid=db.save_run(a["dataset"],a["id"],b["id"],turns,evaluation)
    return {"run_id":rid,"dataset":a["dataset"],"participants":[a["display_name"],b["display_name"]],
            "turns":turns,"evaluation":evaluation}

@app.get("/agents/runs")
def runs(dataset: str="synthetic", participant_id: str|None=None):
    if dataset not in ("synthetic","real"): raise HTTPException(400,"Invalid dataset")
    _require_real_data_enabled(dataset)
    return db.get_runs(dataset, a=participant_id)

@app.post("/rankings/{pid}")
def rankings(pid: str):
    p=db.get_participant(pid)
    if not p: raise HTTPException(404,"Participant not found")
    _require_real_data_enabled(p["dataset"])
    candidates=[x for x in db.list_participants(p["dataset"]) if x["id"]!=pid]
    ranked=[]
    for other in candidates:
        if other["age"]<18: continue
        result=compatibility(p,other)
        result.update({"participant_id":other["id"],"display_name":other["display_name"],"age":other["age"],
                       "linkedin_url":other.get("linkedin_url"),"instagram_url":other.get("instagram_url")})
        ranked.append(result)
    ranked.sort(key=lambda x:x["score"],reverse=True)
    db.save_rankings(p["dataset"],pid,ranked)
    return {"participant_id":pid,"dataset":p["dataset"],"eligible_pairs_evaluated":len(ranked),
            "ranking_method":"transparent overlap heuristic; not scientifically validated","results":ranked}

@app.get("/rankings/{pid}")
def get_rankings(pid: str):
    p=db.get_participant(pid)
    if not p: raise HTTPException(404,"Participant not found")
    _require_real_data_enabled(p["dataset"])
    results=db.get_rankings(p["dataset"],pid)
    if results is None: raise HTTPException(404,"No rankings yet; run POST /rankings/{pid}")
    return {"participant_id":pid,"dataset":p["dataset"],"results":results}

@app.post("/demo/reset")
def reset_demo():
    """Reset only synthetic data. Real participants are never touched."""
    with db.connect() as c:
        c.execute("DELETE FROM participants WHERE dataset='synthetic'")
    from .seed import seed_synthetic
    count=seed_synthetic()
    return {"reset":True,"dataset":"synthetic","participants":count}
