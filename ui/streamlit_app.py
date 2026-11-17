"""Small Streamlit front end that talks to the FastAPI service."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

STATUS_ICON = {"match": "✅", "partial": "🟡", "missing": "❌"}

st.set_page_config(page_title="Job Fit Agent", page_icon="🎯", layout="wide")
st.title("🎯 Job Fit Agent")
st.caption("A LangChain agent that reads a job posting, searches your CV for evidence and reports gaps.")

with st.sidebar:
    st.header("Your CV")
    cv_file = st.file_uploader("Upload PDF / .txt / .md", type=["pdf", "txt", "md"])
    st.caption("Your CV is sent to the configured LLM provider for analysis.")

st.header("Job posting")
tab_url, tab_text = st.tabs(["From URL", "Paste text"])
with tab_url:
    job_url = st.text_input("Public job posting URL", placeholder="https://...")
with tab_text:
    job_text = st.text_area("Paste the posting text", height=220)

if st.button("Analyse fit", type="primary", disabled=cv_file is None or not (job_url or job_text)):
    files = {"cv_file": (cv_file.name, cv_file.getvalue())}
    data = {"job_text": job_text} if job_text else {"job_url": job_url}
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    with st.spinner("The agent is reading the posting and searching your CV..."):
        try:
            resp = requests.post(f"{API_URL}/analyze", files=files, data=data, headers=headers, timeout=180)
        except requests.RequestException:
            st.error(f"Could not reach the API at {API_URL}.")
            st.stop()

    if resp.status_code != 200:
        st.error(resp.json().get("detail", f"Request failed ({resp.status_code})."))
        st.stop()

    out = resp.json()
    report = out["report"]

    left, right = st.columns([1, 3])
    left.metric("Fit score", f"{out['score']} / 100")
    right.subheader(f"{report['job_title']}" + (f" · {report['company']}" if report.get("company") else ""))
    right.write(report["summary"])

    st.subheader("Requirements")
    st.dataframe(
        [
            {
                "": STATUS_ICON[m["status"]],
                "Requirement": m["requirement"],
                "Importance": m["importance"].replace("_", " "),
                "Evidence in CV": m["evidence"] or "—",
            }
            for m in report["matches"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top gaps")
        for gap in report["top_gaps"]:
            st.markdown(f"- {gap}")
    with col_b:
        st.subheader("CV suggestions")
        for tip in report["cv_suggestions"]:
            st.markdown(f"- {tip}")

    with st.expander(f"Agent steps ({len(out['steps'])} tool calls) · model: {out['model']}"):
        for i, step in enumerate(out["steps"], 1):
            st.code(f"{i}. {step['tool']}({step['args']})", language="text")
