from __future__ import annotations
import json
from pathlib import Path
from uuid import uuid4
import streamlit as st
from forgeai.pipeline import build_agent
from forgeai.runner import from_definition

st.set_page_config(page_title="ForgeAI Agent Factory", page_icon="🌊", layout="wide")
st.title("ForgeAI Agent Factory")
st.caption("Build and evaluate document-grounded agents locally. Offline chat is extractive and quotes retrieved evidence.")

build_tab, chat_tab, report_tab = st.tabs(["Agent Factory", "Agent Chat", "Evaluation & Reports"])
with build_tab:
    brief = st.text_area("Agent brief", height=130, placeholder="Describe the agent, its users, scope, and refusal behavior.")
    uploads = st.file_uploader("Provide documents (PDF, TXT, Markdown, CSV)", type=["pdf", "txt", "md", "csv"], accept_multiple_files=True)
    if st.button("Generate, evaluate, and optimize", type="primary", disabled=not brief.strip() or not uploads):
        collection = Path(".forgeai_uploads") / uuid4().hex
        collection.mkdir(parents=True, exist_ok=True)
        for uploaded in uploads:
            name = Path(uploaded.name).name
            if Path(name).suffix.lower() not in {".pdf", ".txt", ".md", ".csv"}: st.error(f"Unsupported file: {name}"); st.stop()
            (collection / name).write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Running the ForgeAI workflow..."):
                result = build_agent(brief, collection, Path("artifacts"))
            st.session_state["active_agent"] = str(Path("artifacts") / result["agent_id"] / "agent_definition.json")
            st.success(f"Built {result['agent_name']} with {result['evaluation']['case_count']} evaluation cases. Final pass rate: {result['final_score']['pass_rate']:.1%}")
            st.json({"requirements": result["requirements"], "groundability": result["groundability"], "optimization_iterations": len(result["optimization_logs"])})
        except Exception as exc: st.error(str(exc))

with chat_tab:
    agent_path = st.text_input("Agent definition", value=st.session_state.get("active_agent", ""))
    if agent_path:
        question = st.text_input("Question")
        if st.button("Ask", disabled=not question.strip()):
            try:
                response = from_definition(agent_path).answer(question)
                st.write(response["answer"])
                if response["sources"]: st.caption("Sources: " + "; ".join(f"{s['source']}" + (f" row {s['row']}" if s["row"] else "") for s in response["sources"]))
                st.caption(f"Provider: {response['provider']}")
            except Exception as exc: st.error(str(exc))

with report_tab:
    report_path = st.text_input("Final report JSON", value=(str(Path(agent_path).with_name("final_report.json")) if agent_path else ""))
    if st.button("Load report", disabled=not report_path.strip()):
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("Cases", report["evaluation"]["case_count"])
            b.metric("Initial pass rate", f"{report['initial_score']['pass_rate']:.1%}")
            c.metric("Final pass rate", f"{report['final_score']['pass_rate']:.1%}")
            st.subheader("Optimization history")
            st.json(report["optimization_logs"])
            st.subheader("Remaining failures")
            st.json(report["final_score"]["failures"])
            markdown = Path(report_path).with_name("final_report.md")
            if markdown.exists(): st.download_button("Download readable report", markdown.read_text(encoding="utf-8"), file_name=markdown.name)
        except Exception as exc: st.error(str(exc))
