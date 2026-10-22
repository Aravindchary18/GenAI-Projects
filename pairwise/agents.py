"""Independent profile-bound agents. Local mode generates responses at runtime from each profile;
optional OpenAI mode uses an LLM provider. This is not a transcript fixture."""
import json, os, re
from collections import Counter

def _tokens(values):
    return {str(x).strip().lower() for x in values if str(x).strip()}

def profile_text(p):
    return " ".join([p.get("bio","")] + p.get("interests",[]) + p.get("hobbies",[]) +
                    p.get("values",[]) + p.get("relationship_goals",[]) + p.get("lifestyle",[])).lower()

def analyze_profile(p):
    """Only summarizes participant-submitted/consented text; no social scraping or identity inference."""
    explicit = {
        "interests": p.get("interests", []),
        "hobbies": p.get("hobbies", []),
        "values": p.get("values", []),
        "relationship_needs": p.get("relationship_goals", []),
        "lifestyle": p.get("lifestyle", []),
    }
    evidence = []
    for field in ("bio","interests","hobbies","values","relationship_goals","lifestyle"):
        val = p.get(field, [])
        if val:
            evidence.append({"field": field, "evidence": val if isinstance(val, str) else ", ".join(val),
                             "source": "participant-submitted form"})
    missing = [k for k,v in explicit.items() if not v]
    analysis = {
        **explicit,
        "evidence": evidence,
        "uncertainties": [f"No participant-provided details for {k.replace('_',' ')}." for k in missing],
        "source_method": "Participant-submitted form fields; no scraping or unapproved inference.",
        "confidence": "limited" if missing else "based on explicit self-report",
        "note": "This is a self-report summary, not a psychological assessment."
    }
    # Optional provider is deliberately opt-in and does not fetch external profiles.
    if os.getenv("OPENAI_API_KEY") and os.getenv("USE_LLM_FOR_ANALYSIS","false").lower()=="true":
        try:
            from openai import OpenAI
            client=OpenAI()
            prompt = ("Return JSON with interests, hobbies, values, relationship_needs, lifestyle, "
                      "evidence, uncertainties. Only use the explicit input, never infer protected traits. "
                      "Keep uncertainties if evidence is absent. Input:\n"+json.dumps(explicit|{"bio":p.get("bio","")}))
            r=client.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"system","content":"You summarize adult dating profiles cautiously. Output valid JSON only."},
                          {"role":"user","content":prompt}], temperature=0.2, response_format={"type":"json_object"})
            parsed=json.loads(r.choices[0].message.content)
            parsed["source_method"]="Participant-submitted form fields summarized by configured LLM; no scraping."
            parsed.setdefault("uncertainties", analysis["uncertainties"])
            parsed.setdefault("evidence", evidence)
            parsed["note"]="LLM-generated self-report summary; not a psychological assessment."
            return parsed
        except Exception as e:
            analysis["provider_warning"] = f"LLM analysis failed; used explicit-form summary instead ({type(e).__name__})."
    return analysis

def _shared(a,b):
    fields=["interests","hobbies","values","relationship_goals","lifestyle"]
    out={}
    for f in fields:
        av=_tokens(a.get(f,[])); bv=_tokens(b.get(f,[]))
        common=sorted(av & bv)
        if common: out[f]=common
    return out

def compatibility(a,b):
    """Transparent heuristic score: not validated, not a prediction of relationship success."""
    shared=_shared(a,b)
    fields=["interests","hobbies","values","relationship_goals","lifestyle"]
    weights={"interests":0.25,"hobbies":0.15,"values":0.25,"relationship_goals":0.25,"lifestyle":0.10}
    per={}
    for f in fields:
        av=_tokens(a.get(f,[])); bv=_tokens(b.get(f,[]))
        if av or bv:
            per[f]=len(av & bv)/max(1,len(av | bv))
        else:
            per[f]=None
    available=[(weights[k],v) for k,v in per.items() if v is not None]
    score=round(100*sum(w*v for w,v in available)/sum(w for w,v in available),1) if available else 0.0
    # Hard constraints are user-authored only.
    conflicts=[]
    pa=a.get("preferences",{}); pb=b.get("preferences",{})
    for owner, target in ((a,b),(b,a)):
        pref=owner.get("preferences",{})
        target_age=target.get("age")
        lo=pref.get("min_age"); hi=pref.get("max_age")
        if target_age is not None and ((lo is not None and target_age < lo) or (hi is not None and target_age > hi)):
            conflicts.append(f"{owner.get('display_name')} specified an age range that excludes the other participant.")
        wanted=set(map(str.lower,pref.get("must_have_interests",[])))
        if wanted and not (wanted & _tokens(target.get("interests",[]))):
            conflicts.append(f"{owner.get('display_name')} listed must-have interests not found in the other profile.")
    return {
        "score": score,
        "score_type":"transparent heuristic, not scientifically validated",
        "shared_interests": shared.get("interests",[]),
        "shared_hobbies": shared.get("hobbies",[]),
        "shared_values": shared.get("values",[]),
        "shared_relationship_goals": shared.get("relationship_goals",[]),
        "shared_lifestyle": shared.get("lifestyle",[]),
        "field_overlap": per,
        "possible_incompatibilities": sorted(set(conflicts)),
        "uncertainties": ["Compatibility reflects only entered self-reported fields.",
                          "Missing or differently worded information can lower apparent overlap.",
                          "A score does not predict attraction, consent, or relationship success."]
    }

def _local_agent_reply(me, other, stage, prior_turns):
    shared=_shared(me,other)
    my_interests=set(me.get("interests",[])+me.get("hobbies",[]))
    other_interests=set(other.get("interests",[])+other.get("hobbies",[]))
    common=sorted(my_interests & other_interests)
    values=sorted(set(me.get("values",[])) & set(other.get("values",[])))
    goals=sorted(set(me.get("relationship_goals",[])) & set(other.get("relationship_goals",[])))
    missing=[x for x in ("interests","values","relationship_goals","lifestyle") if not other.get(x)]
    questions=[]
    if not other.get("relationship_goals"): questions.append("What kind of relationship are you hoping to build?")
    if not other.get("lifestyle"): questions.append("What does a good, ordinary week look like for you?")
    if not other.get("values"): questions.append("Which values matter most to you in a relationship?")
    if not common: questions.append("What hobby or interest would you most enjoy sharing with a partner?")
    if not questions: questions=["Which shared interest would you most like to explore together?"]
    if stage % 2 == 0:
        pieces=[f"I’m {me['display_name']}'s profile-based agent. I’m exploring compatibility with {other['display_name']} using only their shared, self-reported profile details."]
        if common: pieces.append("A possible conversation starter is "+", ".join(common[:3])+".")
        if values: pieces.append("We both explicitly list "+", ".join(values[:2])+" as values.")
        if goals: pieces.append("There may be alignment around "+", ".join(goals[:2])+".")
        if missing: pieces.append("I don’t have enough information about "+", ".join(missing)+", so I won’t assume.")
        pieces.append("Question: "+questions[stage % len(questions)])
        return " ".join(pieces)
    else:
        pieces=["I’m assessing whether this looks like a respectful, mutually interesting conversation—not deciding for either person."]
        if common: pieces.append("The strongest concrete common ground is "+", ".join(common[:3])+".")
        if not values: pieces.append("Values overlap is not evident from the current fields; that is unknown, not necessarily a conflict.")
        if not goals: pieces.append("Relationship-goal alignment is unclear from the available self-reports.")
        pieces.append("A useful next step would be to ask about "+questions[stage % len(questions)].replace("Question: ","").lower())
        return " ".join(pieces)

def run_pair(a,b,max_turns=4):
    """Execute each agent in turn; each message is generated from current profiles and conversation state."""
    turns=[]
    for i in range(max_turns):
        speaker, other = (a,b) if i%2==0 else (b,a)
        if os.getenv("OPENAI_API_KEY") and os.getenv("USE_LLM_AGENTS","false").lower()=="true":
            try:
                from openai import OpenAI
                client=OpenAI()
                prior="\n".join(f"{t['speaker']}: {t['message']}" for t in turns)
                system=("You are the private dating-profile agent for "+speaker["display_name"]+". "
                        "Use only the owner's and other adult participant's explicit profile information. "
                        "Do not impersonate a human, claim consent, infer sensitive traits, pressure anyone, "
                        "or make decisions for participants. Ask thoughtful, respectful compatibility questions. "
                        "Be clear you are an AI agent.")
                user={"your_profile":speaker,"other_profile":other,"conversation_so_far":prior,
                      "task":"Write the next concise agent-to-agent message. Avoid repeating previous messages."}
                r=client.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                    messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(user)}],
                    temperature=0.6)
                msg=r.choices[0].message.content.strip()
                mode="llm"
            except Exception:
                msg=_local_agent_reply(speaker,other,i,turns); mode="local-fallback"
        else:
            msg=_local_agent_reply(speaker,other,i,turns); mode="local-profile-agent"
        turns.append({"turn":i+1,"speaker":speaker["display_name"]+" Agent","speaker_id":speaker["id"],
                      "message":msg,"execution_mode":mode})
    evaluation=compatibility(a,b)
    evaluation["agent_summary"] = (
        f"The agents identified {len(evaluation['shared_interests'])} shared interests, "
        f"{len(evaluation['shared_values'])} shared values, and "
        f"{len(evaluation['possible_incompatibilities'])} explicit preference conflicts. "
        "This is a conversation aid, not a decision about either participant."
    )
    return turns,evaluation
