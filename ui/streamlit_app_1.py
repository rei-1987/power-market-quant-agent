"""Single-screen Streamlit UI connected to the real backend runtime.

No mock router or fake quantitative results are used.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_runtime import AgentRuntime


st.set_page_config(
    page_title="Power-Market Quant Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# One-screen layout
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
/* Hide Streamlit chrome */
[data-testid="stHeader"], [data-testid="stToolbar"] {display:none !important;}
#MainMenu, footer {visibility:hidden !important;}

/* Lock page to one viewport */
html, body, [data-testid="stAppViewContainer"], .stApp {
    height:100vh !important;
    min-height:100vh !important;
    overflow:hidden !important;
    background:#071426 !important;
    color:#EAF2FF !important;
}
[data-testid="stAppViewContainer"] > .main {
    height:100vh !important;
    overflow:hidden !important;
}
.block-container {
    width:100% !important;
    max-width:1900px !important;
    height:100vh !important;
    max-height:100vh !important;
    overflow:hidden !important;
    padding:.48rem .72rem .45rem .72rem !important;
    box-sizing:border-box !important;
}

/* Global spacing */
div[data-testid="stVerticalBlock"] {gap:.34rem !important;}
div[data-testid="stHorizontalBlock"] {gap:.48rem !important;}
p {margin-bottom:.18rem !important;}

/* Header */
.pm-header {
    height:76px;
    background:linear-gradient(180deg,#0B1A31 0%,#08162A 100%);
    border:1px solid #173C6B;
    border-radius:14px;
    padding:11px 18px;
    margin-bottom:7px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    box-sizing:border-box;
}
.pm-title {
    color:#F4F8FF;
    font-size:1.65rem;
    line-height:1.05;
    font-weight:850;
    letter-spacing:-.02em;
}
.pm-subtitle {
    color:#8FA7C9;
    font-size:.76rem;
    margin-top:5px;
}
.pm-badge {
    padding:7px 13px;
    border-radius:999px;
    border:1px solid #2A5EA4;
    background:#0A2750;
    color:#82B8FF;
    font-size:.72rem;
    font-weight:800;
    white-space:nowrap;
}

/* Main cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(180deg,#0B1B34 0%,#09192F 100%) !important;
    border:1px solid #17467C !important;
    border-radius:13px !important;
    box-shadow:none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding:.68rem .72rem !important;
}
.panel-title {
    font-size:.94rem;
    font-weight:850;
    color:#F2F7FF;
    margin-bottom:.35rem;
}
.muted {color:#7F96B9;font-size:.67rem;line-height:1.3;}

/* Left */
.market {
    height:38px;
    display:flex;
    align-items:center;
    gap:8px;
    padding:0 10px;
    border-radius:9px;
    background:#08172B;
    border:1px solid #245B96;
    color:#EEF5FF;
    font-size:.78rem;
    font-weight:800;
    margin-bottom:4px;
}
.flag {
    width:19px;height:12px;border-radius:2px;
    background:linear-gradient(#111 0 33%,#D00000 33% 66%,#FFCE00 66% 100%);
}
label[data-testid="stWidgetLabel"] p {
    color:#AFC3E1 !important;
    font-size:.69rem !important;
    font-weight:800 !important;
}
div[data-testid="stTextArea"] textarea {
    background:#08172B !important;
    color:#E7F0FF !important;
    border:1px solid #2A5D97 !important;
    border-radius:10px !important;
    min-height:112px !important;
    height:112px !important;
    font-size:.77rem !important;
    line-height:1.34 !important;
    padding:.60rem !important;
}
.stButton > button {
    min-height:1.95rem !important;
    border-radius:9px !important;
    font-size:.72rem !important;
    font-weight:800 !important;
    padding:.15rem .35rem !important;
    background:#0B1B34 !important;
    color:#D8E7FF !important;
    border:1px solid #2B5F9A !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(180deg,#4F6EFF,#3155D4) !important;
    color:#FFF !important;
    border:1px solid #6281FF !important;
}
.mode-pill {
    display:inline-block;
    padding:5px 9px;
    border-radius:999px;
    background:#0A2C59;
    border:1px solid #2A75CC;
    color:#80BDFF;
    font-size:.68rem;
    font-weight:850;
}

/* Center */
.status-box {
    background:#08172B;
    border:1px solid #16416F;
    border-radius:10px;
    padding:9px 11px;
    color:#D8E6FA;
    font-size:.75rem;
    line-height:1.34;
}
.spec-grid {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:7px;
}
.spec-item {
    background:#08172B;
    border:1px solid #16416F;
    border-radius:9px;
    padding:8px;
    min-height:54px;
}
.spec-key {
    font-size:.57rem;
    color:#7F96B9;
    text-transform:uppercase;
    font-weight:850;
}
.spec-val {
    font-size:.71rem;
    color:#F0F5FF;
    margin-top:3px;
    line-height:1.25;
    overflow-wrap:anywhere;
}
div[data-testid="stAlert"] {
    padding:.45rem .6rem !important;
    border-radius:9px !important;
    font-size:.70rem !important;
}

/* Right activity */
.event {
    background:#08172B;
    border:1px solid #16416F;
    border-left:4px solid #4D82FF;
    border-radius:9px;
    padding:6px 8px;
    margin-bottom:5px;
}
.event-label {font-size:.58rem;font-weight:900;color:#83B8FF;}
.event-text {font-size:.68rem;line-height:1.28;color:#D8E6FA;margin-top:2px;}
div[data-testid="stCodeBlock"] {
    border-radius:9px !important;
    overflow:hidden !important;
}
div[data-testid="stCodeBlock"] pre {
    font-size:.57rem !important;
    line-height:1.18 !important;
    max-height:145px !important;
    overflow:hidden !important;
    background:#061225 !important;
}

/* Force the three main columns to the same viewport height */
.main-row {
    height:calc(100vh - 92px);
    overflow:hidden;
}

/* Captions */
.stCaption {font-size:.61rem !important;color:#7489A9 !important;}
</style>
""",
    unsafe_allow_html=True,
)


def ensure_runtime() -> AgentRuntime:
    if "agent_runtime" not in st.session_state:
        st.session_state.agent_runtime = AgentRuntime()
    return st.session_state.agent_runtime


def render_specification(agent_result) -> None:
    spec = agent_result.specification
    validation = agent_result.validation

    st.markdown('<div class="panel-title">Compiled Research Specification</div>', unsafe_allow_html=True)

    values = [
        ("Status", getattr(spec.status, "value", spec.status)),
        ("Output type", getattr(spec.output_type, "value", spec.output_type) if spec.output_type else "—"),
        ("Target", spec.target or "—"),
        ("Reference", spec.reference or "—"),
        ("Study period", str(spec.study_period) if spec.study_period else "—"),
        ("Delivery window", str(spec.delivery_window) if spec.delivery_window else "—"),
    ]

    html = '<div class="spec-grid">'
    for key, val in values:
        html += (
            '<div class="spec-item">'
            f'<div class="spec-key">{key}</div>'
            f'<div class="spec-val">{val}</div>'
            '</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        f"<div class='muted' style='margin-top:6px;'>"
        f"Validation: {'passed' if validation.is_valid else 'failed'}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if validation.errors:
        for error in validation.errors[:3]:
            st.error(error)


def render_events(events) -> None:
    st.markdown('<div class="panel-title">Agent Activity</div>', unsafe_allow_html=True)
    if not events:
        st.markdown('<div class="muted">No backend activity yet.</div>', unsafe_allow_html=True)
        return
    for event in events[:7]:
        st.markdown(
            f"""
<div class="event">
  <div class="event-label">{event.label} · {event.status}</div>
  <div class="event-text">{event.message}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# Header
st.markdown(
    """
<div class="pm-header">
  <div>
    <div class="pm-title">⚡ Power-Market Quant Research Agent</div>
    <div class="pm-subtitle">Natural-language request → real LLM routing → compiler/validator → real data boundary</div>
  </div>
  <div class="pm-badge">Hackathon Demo UI</div>
</div>
""",
    unsafe_allow_html=True,
)

runtime = ensure_runtime()
left, center, right = st.columns([2.25, 7.0, 2.85], gap="small")

# LEFT
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Research Request</div>', unsafe_allow_html=True)
        st.markdown("**Market / Region**")
        st.markdown('<div class="market"><span class="flag"></span>Germany (DE-LU)</div>', unsafe_allow_html=True)

        request = st.text_area(
            "Natural-language request",
            key="request_text",
            placeholder="Ask a research, replay, or live-market question...",
        )

        c_run, c_reset = st.columns(2, gap="small")
        if c_run.button("▶ Run Agent", type="primary", use_container_width=True):
            try:
                st.session_state.runtime_result = runtime.run(request)
                st.session_state.runtime_error = None
            except Exception as exc:
                st.session_state.runtime_result = None
                st.session_state.runtime_error = f"{type(exc).__name__}: {exc}"

        if c_reset.button("↻ Reset", use_container_width=True):
            st.session_state.agent_runtime = AgentRuntime()
            st.session_state.runtime_result = None
            st.session_state.runtime_error = None
            st.rerun()

        result = st.session_state.get("runtime_result")
        error = st.session_state.get("runtime_error")

        st.markdown("**Detected mode (LLM)**")
        if result is not None:
            st.markdown(f'<span class="mode-pill">{result.mode.value}</span>', unsafe_allow_html=True)
            st.caption(result.route_reason)
        elif error:
            st.error(error)
        else:
            st.markdown('<div class="muted">Waiting for the real router.</div>', unsafe_allow_html=True)

        st.markdown("**Demo examples**")
        a, b, c = st.columns([1.25, 1.0, .75], gap="small")
        if a.button("Research", key="demo_research", use_container_width=True):
            st.session_state.request_text = (
                "Do downward total wind forecast revisions make ID1 more likely "
                "to exceed ID3 for evening delivery products over the last 1 year?"
            )
            st.rerun()
        if b.button("Replay", key="demo_replay", use_container_width=True):
            st.session_state.request_text = (
                "Replay 2025-06-12 from 08:00 and show what the agent would have known."
            )
            st.rerun()
        if c.button("Live", key="demo_live", use_container_width=True):
            st.session_state.request_text = (
                "What does the latest wind forecast revision imply for tonight's intraday market?"
            )
            st.rerun()

# CENTER
with center:
    with st.container(border=True):
        result = st.session_state.get("runtime_result")
        error = st.session_state.get("runtime_error")

        st.markdown('<div class="panel-title">Backend Result</div>', unsafe_allow_html=True)

        if error:
            st.error(error)
        elif result is None:
            st.markdown(
                '<div class="status-box">Ready. Submit a request to run the real backend.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
<div class="spec-grid" style="margin-top:8px;">
  <div class="spec-item"><div class="spec-key">Router</div><div class="spec-val">OpenAI LLM</div></div>
  <div class="spec-item"><div class="spec-key">Compiler</div><div class="spec-val">Research compiler</div></div>
  <div class="spec-item"><div class="spec-key">Validator</div><div class="spec-val">Deterministic</div></div>
  <div class="spec-item"><div class="spec-key">Research</div><div class="spec-val">Connected</div></div>
  <div class="spec-item"><div class="spec-key">Replay</div><div class="spec-val">Route only</div></div>
  <div class="spec-item"><div class="spec-key">Live</div><div class="spec-val">Route only</div></div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-box"><b>{result.status}</b><br>{result.message}</div>',
                unsafe_allow_html=True,
            )

            if result.agent_result is not None:
                st.write("")
                render_specification(result.agent_result)

            if result.data_summary:
                st.markdown('<div class="panel-title" style="margin-top:8px;">Real Data Load</div>', unsafe_allow_html=True)
                summary = result.data_summary
                st.markdown(
                    f"""
<div class="spec-grid">
  <div class="spec-item"><div class="spec-key">Rows loaded</div><div class="spec-val">{summary.get('rows_loaded','—')}</div></div>
  <div class="spec-item"><div class="spec-key">Required columns</div><div class="spec-val">{len(summary.get('required_columns',[]))}</div></div>
  <div class="spec-item"><div class="spec-key">Source</div><div class="spec-val">{summary.get('path','—')}</div></div>
</div>
""",
                    unsafe_allow_html=True,
                )

            if result.status == "pipeline_not_implemented":
                st.warning("Routing is real; this mode's dedicated backend pipeline is not implemented yet.")
            elif result.status == "ready_for_quant_test":
                st.info("Next unfinished backend boundary: tools/hypothesis_tests.py")

        if runtime.pending_clarifications:
            st.markdown('<div class="panel-title" style="margin-top:8px;">Clarification Required</div>', unsafe_allow_html=True)
            for req in runtime.pending_clarifications[:3]:
                st.markdown(f"**{req.question}**")
                if req.options:
                    st.caption(" · ".join(opt.label for opt in req.options))
            answer = st.text_input(
                "Clarification answer",
                placeholder="Example: Total wind revision, last 1 year",
            )
            if st.button("Submit clarification"):
                try:
                    st.session_state.runtime_result = runtime.answer_clarification(answer)
                    st.session_state.runtime_error = None
                    st.rerun()
                except Exception as exc:
                    st.session_state.runtime_error = f"{type(exc).__name__}: {exc}"
                    st.rerun()

# RIGHT
with right:
    result = st.session_state.get("runtime_result")

    with st.container(border=True):
        render_events(result.events if result is not None else [])

    with st.container(border=True):
        st.markdown('<div class="panel-title">Latest Backend Output</div>', unsafe_allow_html=True)
        if result is None:
            st.code('{"status":"waiting"}', language="json")
        else:
            st.code(json.dumps(result.to_dict(), indent=2, default=str), language="json")
