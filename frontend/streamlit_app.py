import requests
import streamlit as st

FASTAPI_URL = "http://backend:8000"

st.set_page_config(
    page_title="Unified AI",
    layout = "wide"
)

st.title("Unified AI")

st.caption(
    "AI-generated results may contain mistakes. "
    "Please verify information with reliable sources."
)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Chat",
        "Resume Analysis",
        "Skill Gap",
        "Career roadmap",
        "Web Search"
    ]
)

with tab1:
    
    st.subheader("Chat with Documents")
    
    document_file = st.file_uploader(
        "Upload Document PDF",
        type=["pdf"],
        key="document"
    )

    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False

    if "document_id" not in st.session_state:
        st.session_state.document_id = None 

    if st.button("Upload document"):
        if document_file:

            try:
                    files = {
                        "file": (
                            document_file.name,
                            document_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    response = requests.post(
                        f"{FASTAPI_URL}/upload",
                        files=files,
                        timeout=600
                    )

                    response.raise_for_status()

                    data = response.json()

                    st.session_state.document_uploaded = True

                    st.session_state.document_id = data["document_id"]


            except Exception as e :
                  st.error(
                      f"Upload failed: {e}"
                  )
        else:
            st.info(
                "Please upload a PDF first."
            )

    if st.session_state.document_uploaded:
            question = st.text_area(
            "Ask question"
            )

            if st.button("Send"):

                if question.strip():
                    try :
                        response = requests.post(
                            f"{FASTAPI_URL}/chat",
                            json={
                                "question": question,
                                "document_id": st.session_state.document_id
                            },
                            stream=True,
                            timeout=600
                        )
                        response.raise_for_status()
                        answer = ""
                        placeholder= st.empty()

                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk :
                                answer+=chunk
                                placeholder.markdown(answer)
                    except Exception as e:
                        st.error(
                            f"Chat failed: {e}"
                        )

                else:
                    st.info("Please enter the question.")
with tab2:     

    st.subheader("Resume Analysis")

    resume_file = st.file_uploader(
        "Upload a resume PDF only.",
        type=["pdf"],
        key="resume"
    )

    if st.button("Analyze Resume"):

        if resume_file:

            try:

                files = {
                    "file": (
                        resume_file.name,
                        resume_file.getvalue(),
                        "application/pdf"
                    )
                }

                response = requests.post(
                    f"{FASTAPI_URL}/career_mentor/analyze",
                    files=files,
                    timeout=600
                )

                response.raise_for_status()

                st.json(
                    response.json()
                )

            except Exception as e :

                st.error(
                    f"Resume analysis failed: {e}"
                )

with tab3:

    st.subheader("Skill Gap Analysis")

    skill_file = st.file_uploader(
        "Upload a resume PDF only.",
        type =["pdf"],
        key="skillgap"
    )

    job_description = st.text_area(
        "Paste Job Description"
    )

    if st.button("Analysis Skill Gap"):

        if skill_file and job_description.strip():

            try:

                files = {
                    "file": (
                        skill_file.name,
                        skill_file.getvalue(),
                        "application/pdf"
                    )
                }

                data = {
                    "job_description": job_description
                }

                response = requests.post(
                    f"{FASTAPI_URL}/career_mentor/skill-gap",
                    files=files,
                    data=data,
                    timeout=600
                )

                response.raise_for_status()

                st.json(
                    response.json()
                )

            except Exception as e:

                st.error(
                    f"Skill gap failed: {e}"
                )

with tab4:

    st.subheader("Career Roadmap")

    current_skills = st.text_input(
        "Current Skills"
    )

    target_role = st.text_input(
        "Target Role"
    )

    experience_years = st.number_input(
        "Experience Years",
        min_value=0,
        max_value=50,
        value=0 
    )

    if st.button("Generate Roadmap"):

        if current_skills.strip() and target_role.strip():

            try:

                data = {
                    "current_skills": current_skills,
                    "target_role": target_role,
                    "experience_years": int(experience_years)
                }

                response = requests.post(
                    f"{FASTAPI_URL}/career_mentor/roadmap",
                    data=data,
                    timeout=600
                )

                response.raise_for_status()

                st.json(
                    response.json()
                )

            except Exception as e :

                st.error(
                    f"Roadmap failed: {e}"
                )

with tab5:

    st.subheader("Web Search")

    search_question = st.text_area(
        "Ask a web search question"
    )

    if st.button("Search Web"):

        if search_question.strip():

            try:

                response = requests.post(
                    f"{FASTAPI_URL}/websearch/search",
                    json={
                        "question": search_question
                    },
                    timeout=600
                )

                response.raise_for_status()

                result = response.json()

                st.markdown(
                    result["result"]
                )

            except Exception as e:

                st.error(
                    f"Web search failed: {e}"
                )

        else:

            st.info(
                "Please enter a question."
            )