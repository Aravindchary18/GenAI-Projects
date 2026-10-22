import json, os, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = os.getenv("DATING_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "dating.sqlite3"))

def now():
    return datetime.now(timezone.utc).isoformat()

def connect():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c

def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS participants(
          id TEXT PRIMARY KEY, dataset TEXT NOT NULL, display_name TEXT NOT NULL, age INTEGER NOT NULL,
          bio TEXT NOT NULL, interests TEXT NOT NULL, hobbies TEXT NOT NULL, values_json TEXT NOT NULL,
          relationship_goals TEXT NOT NULL, lifestyle TEXT NOT NULL, preferences TEXT NOT NULL,
          linkedin_url TEXT, instagram_url TEXT, consent_confirmed INTEGER NOT NULL DEFAULT 0,
          source_text TEXT, analysis TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_runs(
          id TEXT PRIMARY KEY, dataset TEXT NOT NULL, participant_a_id TEXT NOT NULL,
          participant_b_id TEXT NOT NULL, turns TEXT NOT NULL, evaluation TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(participant_a_id) REFERENCES participants(id) ON DELETE CASCADE,
          FOREIGN KEY(participant_b_id) REFERENCES participants(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS rankings(
          dataset TEXT NOT NULL, participant_id TEXT NOT NULL, results TEXT NOT NULL,
          created_at TEXT NOT NULL, PRIMARY KEY(dataset, participant_id),
          FOREIGN KEY(participant_id) REFERENCES participants(id) ON DELETE CASCADE
        );
        """)

def row_to_participant(r):
    if not r: return None
    d = dict(r)
    for col, key in [("interests","interests"),("hobbies","hobbies"),("values_json","values"),
                     ("relationship_goals","relationship_goals"),("lifestyle","lifestyle"),
                     ("preferences","preferences"),("analysis","analysis")]:
        raw = d.pop(col, None)
        d[key] = json.loads(raw) if raw else ({} if key in ("preferences","analysis") else [])
    d["consent_confirmed"] = bool(d.get("consent_confirmed"))
    d.pop("source_text", None)  # never return submitted source text in ordinary API payloads
    return d

def get_participant(pid):
    with connect() as c:
        r = c.execute("SELECT * FROM participants WHERE id=?", (pid,)).fetchone()
    return row_to_participant(r)

def list_participants(dataset):
    with connect() as c:
        rows = c.execute("SELECT * FROM participants WHERE dataset=? ORDER BY created_at", (dataset,)).fetchall()
    return [row_to_participant(r) for r in rows]

def insert_participant(p):
    pid = str(uuid.uuid4())
    with connect() as c:
        c.execute("""INSERT INTO participants VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            pid, p["dataset"], p["display_name"], p["age"], p["bio"],
            json.dumps(p.get("interests",[])), json.dumps(p.get("hobbies",[])),
            json.dumps(p.get("values",[])), json.dumps(p.get("relationship_goals",[])),
            json.dumps(p.get("lifestyle",[])), json.dumps(p.get("preferences",{})),
            p.get("linkedin_url"), p.get("instagram_url"), int(p.get("consent_confirmed",False)),
            p.get("source_text"), None, now()
        ))
    return get_participant(pid)

def save_analysis(pid, analysis):
    with connect() as c:
        c.execute("UPDATE participants SET analysis=? WHERE id=?", (json.dumps(analysis), pid))

def delete_participant(pid):
    with connect() as c:
        cur = c.execute("DELETE FROM participants WHERE id=?", (pid,))
        return cur.rowcount > 0

def save_run(dataset, a, b, turns, evaluation):
    rid = str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO agent_runs VALUES (?,?,?,?,?,?,?)",
                  (rid, dataset, a, b, json.dumps(turns), json.dumps(evaluation), now()))
    return rid

def get_runs(dataset, a=None, b=None, limit=100):
    sql = "SELECT * FROM agent_runs WHERE dataset=?"
    params = [dataset]
    if a:
        sql += " AND (participant_a_id=? OR participant_b_id=?)"
        params += [a,a]
    if b:
        sql += " AND ((participant_a_id=? AND participant_b_id=?) OR (participant_a_id=? AND participant_b_id=?))"
        params += [a,b,b,a]
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect() as c:
        rows = c.execute(sql, params).fetchall()
    out=[]
    for r in rows:
        d=dict(r); d["turns"]=json.loads(d["turns"]); d["evaluation"]=json.loads(d["evaluation"]); out.append(d)
    return out

def save_rankings(dataset, pid, results):
    with connect() as c:
        c.execute("INSERT INTO rankings VALUES (?,?,?,?) ON CONFLICT(dataset,participant_id) DO UPDATE SET results=excluded.results, created_at=excluded.created_at",
                  (dataset,pid,json.dumps(results),now()))

def get_rankings(dataset, pid):
    with connect() as c:
        r=c.execute("SELECT results FROM rankings WHERE dataset=? AND participant_id=?",(dataset,pid)).fetchone()
    return json.loads(r["results"]) if r else None
