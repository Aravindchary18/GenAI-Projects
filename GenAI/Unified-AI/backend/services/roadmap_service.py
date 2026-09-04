import requests

import json

def analyze_roadmap(current_skills, target_role, experience_years):
    skills = ", ".join(current_skills) if isinstance(current_skills,(list,tuple)) else str(current_skills)

    prompt = f"""

You are an expert career mentor.

Create a personalized career roadmap based on the user's current stage.

User Details:
- Current Skills: {skills}
- Target Role: {target_role}.
- Experience: {experience_years} years.

Rules:
- Use the EXACT phase names: Foundation, Intermediate, Job Ready.
- Do not rename keys.
- Do not output markdown.
- Do not output explanations.
- Keep items concise and relevant.
- Make sure the roadmap includes the likely skills for {target_role}.

Output format:

1. Current Skill Assessment.
2. Missing Skills.
3. Immediate Next Steps (0-3 months).
4. Mid-Term Goals (3-12 months).
5. Advanced Direction (if applicable).
6. Project Ideas.
7. Learning Resources.
8. Return ONLY valid JSON.
9. Do not include markdown.
10. Do not include explanations before or after the JSON.

Answer:

{{
  "current_assessment": ["..."],
  "roadmap_phases": [
    {{
      "phase": "Foundation",
      "topics": ["..."],
      "projects": ["..."],
      "goal": "..."
    }},
    {{
      "phase": "Intermediate",
      "topics": ["..."],
      "projects": ["..."],
      "goal": "..."
    }},
    {{
      "phase": "Job Ready",
      "topics": ["..."],
      "projects": ["..."],
      "goal": "..."
    }}
  ],
  "interview_preparation": ["..."],
  "portfolio_projects": ["..."],
  "learning_resources": ["..."]
}}
"""
    
    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "qwen2.5-coder:3b-instruct",
            "prompt": prompt,
            "stream": False,
             "format": "json",

            "options": {
                "temperature": 0
            }
            
        },
        timeout=600,
        
    )
    
   
    result = response.json()["response"]
    result = json.loads(result)
    return result


