from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from voice import transcribe_audio, synthesize_speech

load_dotenv()

FASTAPI_URL = (os.getenv("FASTAPI_URL") or "").rstrip("/")
REQUEST_TIMEOUT = 600


def backend_url(path: str) -> str:
    """Build a backend URL."""

    if not FASTAPI_URL:
        raise RuntimeError(
            "FASTAPI_URL is not configured. "
            "Set it in the frontend .env file."
        )

    return f"{FASTAPI_URL}/{path.lstrip('/')}"


def assistant_token() -> str:
    """Read the server-side assistant API token."""

    token = os.getenv("ASSISTANT_API_TOKEN", "").strip()

    if not token:
        try:
            token = str(
                st.secrets.get("ASSISTANT_API_TOKEN", "")
            ).strip()
        except Exception:
            token = ""

    return token


def assistant_headers() -> dict[str, str]:

    token = assistant_token()

    if not token:
        raise RuntimeError(
            "ASSISTANT_API_TOKEN is missing."
        )

    return {
        "X-Assistant-Token": token
    }


def assistant_api_request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the authenticated assistant API."""

    response = requests.request(
        method=method.upper(),
        url=backend_url(path),
        headers=assistant_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError(
            "The assistant API returned an invalid response."
        )

    return data


def get_rag_ans(ques: str, doc_id: str) -> str:
    """Call the existing streaming document-chat endpoint."""

    response = requests.post(
        backend_url("chat"),
        json={
            "question": ques,
            "document_id": doc_id
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    answer = ""
    output = st.empty()

    for chunk in response.iter_content(
        chunk_size=None,
        decode_unicode=True,
    ):
        if chunk:
            answer += chunk
            output.markdown(answer)

    return answer


def render_assistant_task(
    task: dict[str, Any]
) -> None:
    """Display assistant output and execution details."""

    status = str(
        task.get("task_status") or "unknown"
    )

    st.caption(
        f"Task status: **{status.replace('_', ' ').title()}**"
    )

    answer = task.get("final_response")

    if answer:
        st.markdown(str(answer))

    pending = task.get("pending_approval") or {}

    if status == "awaiting_approval" and pending:

        st.warning(
            "This task is waiting for approval. Review the "
            "exact Gmail draft in your Gmail account before "
            "approving its send. Approval here authorizes "
            "only the draft ID shown below."
        )

        st.write(
            f"**Action:** {pending.get('summary', 'Gmail send')}"
        )

        if pending.get("draft_id"):
            st.code(
                str(pending["draft_id"]),
                language=None,
            )

        if pending.get("expires_at"):
            st.caption(
                f"Approval expires: {pending['expires_at']}"
            )

    tool_history = task.get("tool_history") or []

    if tool_history:

        with st.expander(
            "Tool execution trace",
            expanded=False,
        ):

            for index, record in enumerate(
                tool_history,
                start=1,
            ):

                st.markdown(
                    f"**{index}. "
                    f"{record.get('tool_name', 'Unknown tool')}** "
                    f"— {record.get('status', 'unknown')}"
                )

                summary = record.get("result_summary")

                if summary:
                    st.text(str(summary)[:1200])

    errors = task.get("errors") or []

    if errors:

        with st.expander(
            "Execution notes",
            expanded=False,
        ):

            for error in errors:
                st.error(str(error))


def submit_assistant_task(
    prompt: str,
    selected_document_ids: list[str],
) -> dict[str, Any]:
    """Start a new natural-language assistant task."""

    return assistant_api_request(
        "POST",
        "/assistant/tasks",
        payload={
            "user_request": prompt,
            "selected_document_ids": selected_document_ids,
        },
    )


def decide_and_resume_task(
    task: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    """Record approval and resume the paused graph."""

    thread_id = str(
        task.get("thread_id") or ""
    ).strip()

    pending = task.get("pending_approval") or {}

    approval_id = str(
        pending.get("approval_id") or ""
    ).strip()

    if not thread_id or not approval_id:
        raise ValueError(
            "The task has no valid pending approval."
        )

    # First record the actual user's decision.
    assistant_api_request(
        "POST",
        f"/assistant/approvals/{approval_id}/decision",
        payload={
            "decision": decision
        },
    )

    # Then resume the existing LangGraph checkpoint.
    return assistant_api_request(
        "POST",
        f"/assistant/tasks/{thread_id}/resume",
        payload={
            "acknowledgement": {
                "acknowledged": True
            }
        },
    )


st.set_page_config(
    page_title="Unified AI",
    layout="wide",
)

st.title("Unified AI")

st.caption(
    "AI-generated results may contain mistakes. "
    "Review external actions before approving them."
)

# Session state for assistant task history.
#
# Each submitted prompt starts a new backend task.
# This UI history is not persistent multi-turn LLM memory.

st.session_state.setdefault(
    "assistant_task_history",
    []
)

st.session_state.setdefault(
    "last_assistant_task",
    None
)


# Six tabs: the new assistant is additive.

tab_ai, tab_chat, tab_resume, tab_skill, tab_roadmap, tab_web = st.tabs(
    [
        "AI Assistant",
        "Chat",
        "Resume Analysis",
        "Skill Gap",
        "Career roadmap",
        "Web Search",
    ]
)


# ============================================================
# TAB 1 — NATURAL-LANGUAGE AI ASSISTANT
# ============================================================

with tab_ai:

    st.subheader(
        "Ask the assistant to complete a task"
    )

    st.write(
        "Describe what you want in ordinary English. "
        "The LangGraph assistant can select available MCP "
        "tools, execute the required steps and return a summary. "
        "Sensitive actions may pause for your approval."
    )

    configured_token = bool(
        assistant_token()
    )

    if not configured_token:

        st.info(
            "Set ASSISTANT_API_TOKEN in the Streamlit "
            "server environment or secrets, and set "
            "FASTAPI_URL to your backend address."
        )

    # Make an uploaded document available to the assistant.

    uploaded_document_id = st.session_state.get(
        "document_id"
    )

    include_uploaded_document = False

    if uploaded_document_id:

        include_uploaded_document = st.checkbox(
            "Make my uploaded PDF available to this task",
            value=True,
            key="assistant_include_uploaded_pdf",
        )

    # Display previous task activity.

    for entry in st.session_state.assistant_task_history[-10:]:

        with st.chat_message("user"):
            st.markdown(
                entry.get("request", "")
            )

        with st.chat_message("assistant"):
            render_assistant_task(
                entry.get("result", {})
            )

    prompt = st.chat_input(
        "Example: Research this company and prepare an internship enquiry email...",
        disabled=not configured_token,
        key="assistant_task_input",
    )

    if prompt:

        selected_ids = (
            [str(uploaded_document_id)]
            if (
                include_uploaded_document
                and uploaded_document_id
            )
            else []
        )

        with st.spinner(
            "Your assistant is planning and executing the task..."
        ):

            try:

                result = submit_assistant_task(
                    prompt,
                    selected_ids,
                )

                st.session_state.last_assistant_task = result

                st.session_state.assistant_task_history.append(
                    {
                        "request": prompt,
                        "result": result,
                    }
                )

                st.rerun()

            except requests.HTTPError as exc:

                detail = ""

                if exc.response is not None:
                    detail = exc.response.text[:1500]

                st.error(
                    f"Assistant API request failed: {exc}. {detail}"
                )

            except Exception as exc:

                st.error(
                    f"Could not start assistant task: {exc}"
                )

    # Gmail approval interface.

    last_task = st.session_state.get(
        "last_assistant_task"
    )

    if isinstance(last_task, dict):

        pending = last_task.get(
            "pending_approval"
        ) or {}

        if (
            last_task.get("task_status") == "awaiting_approval"
            and pending.get("approval_id")
        ):

            st.divider()

            st.subheader(
                "Approval required"
            )

            render_assistant_task(last_task)

            col_approve, col_reject = st.columns(2)

            with col_approve:

                approve_clicked = st.button(
                    "Approve this Gmail send",
                    type="primary",
                    key=f"approve_{pending['approval_id']}",
                )

            with col_reject:

                reject_clicked = st.button(
                    "Reject",
                    key=f"reject_{pending['approval_id']}",
                )

            selected_decision = (
                "approved" if approve_clicked
                else "rejected" if reject_clicked
                else None
            )

            if selected_decision:

                with st.spinner(
                    "Recording your decision and resuming the task..."
                ):

                    try:

                        resumed = decide_and_resume_task(
                            last_task,
                            selected_decision,
                        )

                        st.session_state.last_assistant_task = resumed

                        history = st.session_state.assistant_task_history

                        if history:
                            history[-1]["result"] = resumed

                        st.rerun()

                    except requests.HTTPError as exc:

                        detail = (
                            exc.response.text[:1500]
                            if exc.response is not None
                            else ""
                        )

                        st.error(
                            f"Approval/resume failed: {exc}. {detail}"
                        )

                    except Exception as exc:

                        st.error(
                            f"Approval/resume failed: {exc}"
                        )

    if st.button(
        "Clear assistant activity",
        key="clear_assistant_history",
    ):

        st.session_state.assistant_task_history = []
        st.session_state.last_assistant_task = None

        st.rerun()


# ============================================================
# TAB 2 — EXISTING DOCUMENT CHAT AND VOICE
# ============================================================

with tab_chat:

    st.subheader("Chat with Documents")

    document_file = st.file_uploader(
        "Upload Document PDF",
        type=["pdf"],
        key="document",
    )

    st.session_state.setdefault(
        "document_uploaded",
        False,
    )

    st.session_state.setdefault(
        "document_id",
        None,
    )

    if st.button(
        "Upload document",
        key="upload_document",
    ):

        if document_file:

            try:

                files = {
                    "file": (
                        document_file.name,
                        document_file.getvalue(),
                        "application/pdf",
                    )
                }

                response = requests.post(
                    backend_url("upload"),
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                data = response.json()

                st.session_state.document_uploaded = True
                st.session_state.document_id = data["document_id"]

                st.success(
                    "Document uploaded successfully."
                )

            except Exception as exc:

                st.error(
                    f"Upload failed: {exc}"
                )

        else:

            st.info(
                "Please upload a PDF first."
            )

    if st.session_state.document_uploaded:

        question = st.text_area(
            "Ask question",
            key="document_question",
        )

        if st.button(
            "Send",
            key="send_document_question",
        ):

            if question.strip():

                try:

                    get_rag_ans(
                        question,
                        st.session_state.document_id,
                    )

                except Exception as exc:

                    st.error(
                        f"Chat failed: {exc}"
                    )

            else:

                st.info(
                    "Please enter the question."
                )

    st.divider()

    st.subheader(
        "You can now ask using Voice"
    )

    aud_file = st.audio_input(
        "Record your question."
    )

    if st.button(
        "Ask with Voice",
        key="rag_voice",
    ):

        if aud_file is None:

            st.warning(
                "Please record your question."
            )

        elif not st.session_state.get("document_uploaded", False):

            st.warning(
                "Please upload a PDF first."
            )

        else:

            try:

                transcript = transcribe_audio(aud_file)

                if not transcript.strip():
                    st.warning("No speech detected.")
                    st.stop()

                st.write(
                    "**Your question:**",
                    transcript,
                )

                answer = get_rag_ans(
                    transcript,
                    st.session_state.document_id,
                )

                if answer.strip():

                    audio = synthesize_speech(answer)

                    st.audio(
                        audio,
                        format="audio/wav",
                    )

            except Exception as exc:

                st.error(
                    f"Voice function failed: {exc}"
                )


# ============================================================
# TAB 3 — EXISTING RESUME ANALYSIS
# ============================================================

with tab_resume:

    st.subheader("Resume Analysis")

    resume_file = st.file_uploader(
        "Upload a resume PDF only.",
        type=["pdf"],
        key="resume",
    )

    if st.button(
        "Analyze Resume",
        key="analyze_resume",
    ):

        if resume_file:

            try:

                files = {
                    "file": (
                        resume_file.name,
                        resume_file.getvalue(),
                        "application/pdf",
                    )
                }

                response = requests.post(
                    backend_url("career_mentor/analyze"),
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                st.json(
                    response.json()["result"]
                )

            except Exception as exc:

                st.error(
                    f"Resume analysis failed: {exc}"
                )

        else:

            st.info(
                "Please upload a resume PDF."
            )


# ============================================================
# TAB 4 — EXISTING SKILL-GAP ANALYSIS
# ============================================================

with tab_skill:

    st.subheader("Skill Gap Analysis")

    skill_file = st.file_uploader(
        "Upload a resume PDF only.",
        type=["pdf"],
        key="skillgap",
    )

    job_description = st.text_area(
        "Paste Job Description",
        key="skillgap_job_description",
    )

    if st.button(
        "Analyse Skill Gap",
        key="analyse_skill_gap",
    ):

        if skill_file and job_description.strip():

            try:

                files = {
                    "file": (
                        skill_file.name,
                        skill_file.getvalue(),
                        "application/pdf",
                    )
                }

                data = {
                    "job_description": job_description
                }

                response = requests.post(
                    backend_url("career_mentor/skill-gap"),
                    files=files,
                    data=data,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                st.json(
                    response.json()["result"]
                )

            except Exception as exc:

                st.error(
                    f"Skill gap failed: {exc}"
                )

        else:

            st.info(
                "Upload a resume and enter a job description."
            )


# ============================================================
# TAB 5 — EXISTING CAREER ROADMAP
# ============================================================

with tab_roadmap:

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
        value=0,
    )

    if st.button(
        "Generate Roadmap",
        key="generate_roadmap",
    ):

        if current_skills.strip() and target_role.strip():

            try:

                data = {
                    "current_skills": current_skills,
                    "target_role": target_role,
                    "experience_years": int(experience_years),
                }

                response = requests.post(
                    backend_url("career_mentor/roadmap"),
                    data=data,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                st.json(
                    response.json()["result"]
                )

            except Exception as exc:

                st.error(
                    f"Roadmap failed: {exc}"
                )

        else:

            st.info(
                "Enter your current skills and target role."
            )


# ============================================================
# TAB 6 — EXISTING WEB SEARCH AND VOICE
# ============================================================

with tab_web:

    st.subheader("Web Search")

    search_question = st.text_area(
        "Ask a web search question",
        key="web_search_question",
    )

    if st.button(
        "Search Web",
        key="search_web",
    ):

        if search_question.strip():

            try:

                response = requests.post(
                    backend_url("websearch/search"),
                    json={
                        "question": search_question
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                st.markdown(
                    response.json()["result"]
                )

            except Exception as exc:

                st.error(
                    f"Web search failed: {exc}"
                )

        else:

            st.info(
                "Please enter a search question."
            )

    st.divider()

    st.subheader(
        "You can now search using voice"
    )

    aud_file = st.audio_input(
        "Record your search question."
    )

    if st.button(
        "Search with voice",
        key="web_voice",
    ):

        if aud_file is None:

            st.warning(
                "Please record your question."
            )

        else:

            try:

                transcript = transcribe_audio(aud_file)

                if not transcript.strip():
                    st.warning("No speech detected.")
                    st.stop()

                st.write(
                    "**Your question:**",
                    transcript,
                )

                response = requests.post(
                    backend_url("websearch/search"),
                    json={
                        "question": transcript
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                result = response.json()["result"]

                st.markdown(result)

                if result.strip():

                    audio = synthesize_speech(result)

                    st.audio(
                        audio,
                        format="audio/wav",
                    )

            except Exception as exc:

                st.error(
                    f"Voice search failed: {exc}"
                )