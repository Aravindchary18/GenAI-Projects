import streamlit as st
import requests, os, sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from app import db
from app.seed import seed_synthetic
from app.agents import analyze_profile, run_pair, compatibility

db.init_db()
API=os.getenv("DATING_API_URL","http://127.0.0.1:8000")
st.set_page_config(page_title="Pairwise · Agentic Dating", page_icon="💫", layout="wide")
st.markdown("""
<style>
:root {color-scheme:light}
.block-container{padding-top:1.4rem;max-width:1400px}
.hero{padding:28px;border-radius:22px;background:linear-gradient(120deg,#211c45,#5a3d91 55%,#d86b9a);color:white;margin-bottom:18px}
.hero h1{font-size:2.5rem;margin-bottom:6px}
.pill{display:inline-block;padding:5px 10px;border-radius:99px;background:#f0eaff;color:#4b2e83;font-size:12px;margin-right:5px}
.metric{padding:15px;border:1px solid #e9e5f2;border-radius:16px;background:#fff}
div[data-testid="stMetric"]{background:#fbf9ff;border:1px solid #eee7fa;padding:12px;border-radius:14px}
div.stButton>button{border-radius:12px}
</style>
<div class="hero"><span class="pill">CONSENT-FIRST</span><span class="pill">AGENTIC MATCHMAKING</span>
<h1>Pairwise 💫</h1><p>Two profile-bound agents. Real-time reasoning. Transparent compatibility—not a verdict.</p></div>
""",unsafe_allow_html=True)

real_data_enabled = os.getenv("ENABLE_REAL_DATA", "false").lower() == "true"
if "dataset" not in st.session_state: st.session_state.dataset="synthetic"
if not real_data_enabled: st.session_state.dataset = "synthetic"
with st.sidebar:
    st.title("Workspace")
    dataset_options = ["synthetic", "real"] if real_data_enabled else ["synthetic"]
    ds=st.radio("Dataset",dataset_options,index=0 if st.session_state.dataset=="synthetic" else 1,
                format_func=lambda x:"🧪 Synthetic demo" if x=="synthetic" else "🔐 Consenting adults")
    if not real_data_enabled:
        st.caption("Real-participant mode is disabled by default. Enable it only for a controlled environment with access controls.")
    st.session_state.dataset=ds
    st.caption("Synthetic and real records are isolated. Demo reset never deletes real records.")
    if st.button("↻ Reset synthetic demo",use_container_width=True):
        with db.connect() as c: c.execute("DELETE FROM participants WHERE dataset='synthetic'")
        seed_synthetic(); st.success("Synthetic demo reset."); st.rerun()
    page=st.radio("Navigate",["Overview","Participants","Profile analysis","Live agent room","Rankings","Privacy & data access"])
    st.divider()
    st.caption("Local SQLite persistence • optional LLM provider")

people=db.list_participants(ds)
if ds=="synthetic" and not people:
    seed_synthetic(); people=db.list_participants(ds)

def getp(pid): return db.get_participant(pid)
def label(p): return f"{p['display_name']} · {p['age']}"
def make_analysis(p):
    a=analyze_profile(p); db.save_analysis(p["id"],a); return a

if page=="Overview":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Participants",len(people))
    c2.metric("Possible pairs",len(people)*(len(people)-1)//2)
    c3.metric("Agent runs",len(db.get_runs(ds)))
    c4.metric("Dataset","Synthetic" if ds=="synthetic" else "Real / consented")
    st.subheader("Explore the workspace")
    cols=st.columns(3)
    cards=[("🧑‍🤝‍🧑","Participant profiles","Create, edit through deletion/recreation, inspect source links and explicit preferences."),
           ("🤖","Live agent room","Run two separate profile-bound agents and watch newly generated turns appear."),
           ("📊","Explainable rankings","Compare every other eligible adult profile with visible overlap and uncertainty.")]
    for col,(emoji,title,desc) in zip(cols,cards):
        with col:
            st.markdown(f"### {emoji} {title}")
            st.write(desc)
    st.info("Scores are heuristic overlap indicators—not validated predictions, and never a substitute for mutual consent.")
    if ds=="real":
        st.warning("Only add adults who explicitly consent to this demonstration. URLs are references only; this app does not scrape LinkedIn or Instagram.")
elif page=="Participants":
    st.subheader("Participants")
    with st.expander("＋ Add a participant",expanded=not people):
        with st.form("new_participant"):
            name=st.text_input("Display name")
            age=st.number_input("Age (adults only)",min_value=18,max_value=100,value=24)
            bio=st.text_area("About me",placeholder="Use information the participant wrote or approved.")
            interests=st.text_input("Interests (comma-separated)")
            hobbies=st.text_input("Hobbies (comma-separated)")
            values=st.text_input("Values (comma-separated)")
            goals=st.text_input("Relationship goals (comma-separated)")
            lifestyle=st.text_input("Lifestyle (comma-separated)")
            linkedin=st.text_input("LinkedIn profile URL (optional)")
            instagram=st.text_input("Public Instagram profile URL (optional)")
            prefs=st.text_input("Must-have interests (comma-separated, optional)")
            consent=st.checkbox("I confirm this adult participant explicitly approved this information and its use in this demo.")
            submit=st.form_submit_button("Create participant",type="primary")
        if submit:
            if not name.strip() or len(bio.strip())<10: st.error("Add a display name and at least 10 characters of bio.")
            elif ds=="real" and not consent: st.error("Consent confirmation is required for the real dataset.")
            elif ds=="real" and not (linkedin.startswith("https://www.linkedin.com/") or linkedin.startswith("https://linkedin.com/") or instagram.startswith("https://www.instagram.com/") or instagram.startswith("https://instagram.com/")):
                st.error("For the real dataset, provide a valid-looking HTTPS LinkedIn or Instagram profile URL.")
            else:
                p={"dataset":ds,"display_name":name.strip(),"age":int(age),"bio":bio.strip(),
                   "interests":[x.strip() for x in interests.split(",") if x.strip()],
                   "hobbies":[x.strip() for x in hobbies.split(",") if x.strip()],
                   "values":[x.strip() for x in values.split(",") if x.strip()],
                   "relationship_goals":[x.strip() for x in goals.split(",") if x.strip()],
                   "lifestyle":[x.strip() for x in lifestyle.split(",") if x.strip()],
                   "preferences":{"must_have_interests":[x.strip() for x in prefs.split(",") if x.strip()]},
                   "linkedin_url":linkedin.strip() or None,"instagram_url":instagram.strip() or None,
                   "consent_confirmed":consent,"source_text":None}
                try:
                    from app.models import ParticipantCreate
                    parsed=ParticipantCreate(**p)
                    created=db.insert_participant(parsed.model_dump())
                    st.success(f"Created {created['display_name']}.")
                    st.rerun()
                except Exception as e: st.error(f"Could not create participant: {e}")
    if not people: st.info("No participants yet.")
    for p in people:
        with st.container(border=True):
            left,right=st.columns([5,1])
            with left:
                st.markdown(f"### {p['display_name']} <span class='pill'>{p['age']} years</span>",unsafe_allow_html=True)
                st.write(p["bio"])
                st.caption("Interests: "+(", ".join(p["interests"]) or "not provided"))
                links=[]
                if p.get("linkedin_url"): links.append(f"[LinkedIn]({p['linkedin_url']})")
                if p.get("instagram_url"): links.append(f"[Instagram]({p['instagram_url']})")
                st.markdown(" · ".join(links) if links else "_No external profile link provided._")
            with right:
                if st.button("Delete",key="del_"+p["id"]):
                    db.delete_participant(p["id"]); st.success("Participant and linked runs/rankings deleted."); st.rerun()
elif page=="Profile analysis":
    st.subheader("Evidence-backed profile analysis")
    if not people: st.info("Create a participant first.")
    else:
        p=st.selectbox("Choose participant",people,format_func=label,key="analysis_person")
        if st.button("Analyze profile",type="primary"):
            st.session_state.analysis_result=make_analysis(p)
        result=p.get("analysis") or st.session_state.get("analysis_result")
        if result:
            st.caption(result.get("source_method",""))
            st.info(result.get("note",""))
            for k in ["interests","hobbies","values","relationship_needs","lifestyle"]:
                st.markdown("**"+k.replace("_"," ").title()+"**")
                st.write(result.get(k,[]) or "Not provided")
            st.markdown("**Evidence**")
            for e in result.get("evidence",[]): st.write(f"• {e['field']}: {e['evidence']} — _{e['source']}_")
            st.markdown("**Uncertainties**")
            for u in result.get("uncertainties",[]): st.write("• "+u)
elif page=="Live agent room":
    st.subheader("Live agent-to-agent conversation")
    if len(people)<2: st.info("Add at least two participants.")
    else:
        c1,c2=st.columns(2)
        a=c1.selectbox("Agent A",people,format_func=label,key="agent_a")
        options=[p for p in people if p["id"]!=a["id"]]
        b=c2.selectbox("Agent B",options,format_func=label,key="agent_b")
        turns_n=st.slider("Conversation turns",2,10,4,2)
        if st.button("▶ Run agents",type="primary"):
            with st.spinner("Agents are independently examining profile evidence and generating messages…"):
                turns,evaluation=run_pair(a,b,turns_n)
                rid=db.save_run(ds,a["id"],b["id"],turns,evaluation)
                st.session_state.last_run={"turns":turns,"evaluation":evaluation,"a":a,"b":b,"id":rid}
        run=st.session_state.get("last_run")
        if run:
            st.caption("Run "+run["id"]+" · execution is generated at runtime; local profile-agent mode requires no API key.")
            for turn in run["turns"]:
                with st.chat_message("assistant" if turn["speaker_id"]==run["a"]["id"] else "user",
                                      avatar="🤖" if turn["speaker_id"]==run["a"]["id"] else "🪄"):
                    st.markdown(f"**{turn['speaker']}**")
                    st.write(turn["message"])
                    st.caption(turn["execution_mode"])
            ev=run["evaluation"]
            st.metric("Transparent overlap score",f"{ev['score']}/100")
            st.write(ev["agent_summary"])
            st.write("**Shared interests:**",", ".join(ev["shared_interests"]) or "None identified")
            st.write("**Possible incompatibilities:**",ev["possible_incompatibilities"] or "No explicit conflict found")
            st.write("**Uncertainty:**", " ".join(ev["uncertainties"]))
elif page=="Rankings":
    st.subheader("Complete pairwise rankings")
    if len(people)<2: st.info("At least two participants are required.")
    else:
        p=st.selectbox("Rank matches for",people,format_func=label,key="rank_person")
        if st.button("Evaluate every eligible pair",type="primary"):
            with st.spinner("Evaluating every other participant in this dataset…"):
                results=[]
                for other in people:
                    if other["id"]==p["id"]: continue
                    ev=compatibility(p,other)
                    ev.update({"participant_id":other["id"],"display_name":other["display_name"],"age":other["age"],
                               "linkedin_url":other.get("linkedin_url"),"instagram_url":other.get("instagram_url")})
                    results.append(ev)
                results.sort(key=lambda x:x["score"],reverse=True)
                db.save_rankings(ds,p["id"],results)
                st.session_state.rank_results={"id":p["id"],"results":results}
        saved=db.get_rankings(ds,p["id"])
        state=st.session_state.get("rank_results")
        results=state["results"] if state and state["id"]==p["id"] else saved
        if results is not None:
            st.caption(f"Evaluated {len(results)} eligible pair(s). Score is not scientifically validated.")
            for i,r in enumerate(results,1):
                with st.container(border=True):
                    c1,c2,c3=st.columns([1,4,1])
                    c1.markdown(f"## #{i}")
                    c2.markdown(f"### {r['display_name']} · {r['age']}")
                    c2.write("Shared interests: "+(", ".join(r["shared_interests"]) or "none identified"))
                    c2.write("Shared values: "+(", ".join(r["shared_values"]) or "none identified"))
                    c2.write("Possible incompatibilities: "+("; ".join(r["possible_incompatibilities"]) or "none explicitly found"))
                    with c2.expander("Evidence, field overlap and uncertainty"):
                        st.json({"field_overlap":r["field_overlap"],"uncertainties":r["uncertainties"]})
                        if r.get("linkedin_url"): st.markdown(f"[LinkedIn source]({r['linkedin_url']})")
                        if r.get("instagram_url"): st.markdown(f"[Instagram source]({r['instagram_url']})")
                    c3.metric("Score",f"{r['score']}/100")
elif page=="Privacy & data access":
    st.subheader("Privacy, consent and source policy")
    st.markdown("""
- **No scraping:** LinkedIn/Instagram URLs are user-provided references only. This build does not log in, scrape, crawl, or infer data from either platform.
- **Consent:** Real-dataset creation requires an explicit checkbox confirming adult participant consent. This is an operator attestation, not an independent consent verification service.
- **Profile analysis:** Uses only form fields the participant supplied or approved. Missing fields remain uncertain.
- **Synthetic mode:** 25 clearly synthetic demo profiles; never presented as real people.
- **Deletion:** Deleting a participant removes their record and cascades to associated runs and rankings in SQLite.
- **Scores:** Transparent field-overlap heuristic, not scientifically validated or predictive.
- **Production gaps:** No authentication/authorization, formal consent receipt/versioning, encryption/key management, rate limiting, independent identity verification, moderation/reporting workflow, or managed database setup. Do not expose real participant data publicly until those are implemented and reviewed.
""")
    st.markdown("**Optional LLM provider**")
    st.code("OPENAI_API_KEY=...\\nUSE_LLM_AGENTS=true\\nUSE_LLM_FOR_ANALYSIS=true\\nOPENAI_MODEL=gpt-4o-mini",language="bash")
    st.caption("Without the optional key, profile-bound local agents still generate runtime messages from the current profiles. That is an executable deterministic agent workflow, not a hosted LLM conversation.")
