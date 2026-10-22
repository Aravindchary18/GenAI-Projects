import pytest
from fastapi.testclient import TestClient
from app import db
from app.api import app
from app.seed import seed_synthetic
from app.agents import run_pair, compatibility

client=TestClient(app)

def sample(dataset="synthetic", name="Test One", consent=True, **extra):
    p={"dataset":dataset,"display_name":name,"age":25,"bio":"I enjoy books and thoughtful conversations.",
       "interests":["books","travel"],"hobbies":["reading"],"values":["honesty"],
       "relationship_goals":["long-term"],"lifestyle":["quiet"],"preferences":{},
       "consent_confirmed":consent,"linkedin_url":None,"instagram_url":None,"source_text":None}
    p.update(extra)
    return p

def test_seed_has_25_synthetic_and_no_real():
    assert seed_synthetic()==25
    assert len(db.list_participants("synthetic"))==25
    assert db.list_participants("real")==[]

def test_agents_generate_turns_from_profiles():
    a=db.insert_participant(sample(name="A"))
    b=db.insert_participant(sample(name="B",interests=["books","music"]))
    turns,ev=run_pair(a,b,4)
    assert len(turns)==4
    assert all(t["message"] for t in turns)
    assert any("books" in t["message"].lower() for t in turns)
    assert ev["score"] >= 0

def test_rank_every_other_pair_and_persist():
    seed_synthetic()
    p=db.list_participants("synthetic")[0]
    results=[compatibility(p,x) for x in db.list_participants("synthetic") if x["id"]!=p["id"]]
    assert len(results)==24
    db.save_rankings("synthetic",p["id"],results)
    assert len(db.get_rankings("synthetic",p["id"]))==24

def test_delete_participant_and_runs_cascade():
    a=db.insert_participant(sample(name="A"))
    b=db.insert_participant(sample(name="B"))
    turns,ev=run_pair(a,b,2)
    db.save_run("synthetic",a["id"],b["id"],turns,ev)
    assert db.delete_participant(a["id"])
    assert db.get_participant(a["id"]) is None
    assert db.get_runs("synthetic")==[]

def test_real_requires_consent_and_url():
    bad=sample(dataset="real",consent=False,linkedin_url="https://www.linkedin.com/in/example")
    r=client.post("/participants",json=bad)
    assert r.status_code==400
    no_url=sample(dataset="real",consent=True)
    assert client.post("/participants",json=no_url).status_code==400
    good=sample(dataset="real",consent=True,instagram_url="https://www.instagram.com/example/")
    assert client.post("/participants",json=good).status_code==200

def test_invalid_profile_url():
    bad=sample(linkedin_url="http://fake.example/user")
    assert client.post("/participants",json=bad).status_code==422

def test_cross_dataset_run_rejected():
    a=db.insert_participant(sample(name="S"))
    b=db.insert_participant(sample(dataset="real",name="R",consent_confirmed=True,
                                  linkedin_url="https://www.linkedin.com/in/example"))
    r=client.post("/agents/run",json={"participant_a_id":a["id"],"participant_b_id":b["id"],"max_turns":2})
    assert r.status_code==400

def test_empty_dataset_rank_endpoint_not_found():
    r=client.get("/participants?dataset=real")
    assert r.status_code==200 and r.json()==[]


def test_real_dataset_is_disabled_without_explicit_operator_opt_in(monkeypatch):
    monkeypatch.setenv("ENABLE_REAL_DATA", "false")
    r = client.get("/participants?dataset=real")
    assert r.status_code == 403
    payload = sample(dataset="real", consent_confirmed=True,
                     linkedin_url="https://www.linkedin.com/in/example")
    r = client.post("/participants", json=payload)
    assert r.status_code == 403


def test_health_discloses_no_social_scraping():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["real_social_scraping"] is False
