"""Fixed presentation UI + real backend bridge.

The frontend layout is fixed and does NOT get redesigned by backend state.
Backend only supplies:
- mode
- compiler / validator status
- clarification questions
- required data columns / data-load status
- later quantitative outputs when implemented

No fake statistics or fake charts.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_controller import AgentController
from app.mode_router import route_request
from schemas.hypothesis_schema import ExecutionMode, SpecificationStatus
from tools.data_tools import required_source_columns, load_market_data


st.set_page_config(
    page_title="Power-Market Quant Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# FIXED UI STYLE — same visual framework as before
# ============================================================
st.markdown(
    """
<style>
[data-testid="stHeader"], [data-testid="stToolbar"] {display:none !important;}
#MainMenu, footer {visibility:hidden !important;}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background:#071426 !important;
    color:#EAF2FF !important;
    overflow:hidden !important;
    height:100vh !important;
}

.block-container {
    max-width:1900px;
    height:100vh;
    overflow:hidden !important;
    padding:.55rem .75rem .45rem .75rem !important;
}

div[data-testid="stVerticalBlock"] {gap:.50rem !important;}
div[data-testid="stHorizontalBlock"] {gap:.55rem !important;}

.pm-header {
    height:78px;
    background:linear-gradient(180deg,#0B1A31 0%,#08162A 100%);
    border:1px solid #173C6B;
    border-radius:14px;
    padding:12px 18px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    box-sizing:border-box;
    margin-bottom:7px;
}
.pm-title {
    color:#F4F8FF;
    font-size:1.72rem;
    font-weight:850;
    letter-spacing:-.02em;
    line-height:1.05;
}
.pm-subtitle {
    color:#94A9C8;
    font-size:.80rem;
    margin-top:6px;
}
.pm-badge {
    padding:8px 14px;
    border:1px solid #2B62A9;
    background:#0A2852;
    color:#83B8FF;
    border-radius:999px;
    font-size:.76rem;
    font-weight:800;
    white-space:nowrap;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(180deg,#0B1B34 0%,#09192F 100%) !important;
    border:1px solid #17467C !important;
    border-radius:13px !important;
    box-shadow:none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding:.72rem .78rem !important;
}

.panel-heading {
    display:flex;
    align-items:center;
    gap:8px;
    color:#F2F7FF;
    font-size:.98rem;
    font-weight:850;
    margin-bottom:6px;
}
.panel-icon {
    width:25px;
    height:25px;
    border-radius:8px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    color:white;
    background:linear-gradient(180deg,#4F6EFF,#2744A6);
    font-size:.82rem;
}

.market-label, label[data-testid="stWidgetLabel"] p {
    color:#AFC3E1 !important;
    font-size:.73rem !important;
    font-weight:800 !important;
    margin-bottom:.15rem !important;
}
.market-pill {
    height:42px;
    display:flex;
    align-items:center;
    gap:9px;
    padding:0 11px;
    border-radius:9px;
    background:#08172B;
    border:1px solid #245B96;
    color:#EEF5FF;
    font-size:.82rem;
    font-weight:800;
    margin-bottom:7px;
}
.flag {
    width:20px;
    height:13px;
    border-radius:2px;
    background:linear-gradient(#111 0 33%,#D00000 33% 66%,#FFCE00 66% 100%);
    box-shadow:0 0 0 1px rgba(255,255,255,.12);
}

div[data-testid="stTextArea"] textarea {
    background:#08172B !important;
    color:#E7F0FF !important;
    border:1px solid #2A5D97 !important;
    border-radius:10px !important;
    min-height:126px !important;
    height:126px !important;
    font-size:.82rem !important;
    line-height:1.36 !important;
    padding:.72rem !important;
}

.stButton > button {
    min-height:2.05rem !important;
    border-radius:9px !important;
    font-weight:800 !important;
    font-size:.78rem !important;
}
.stButton > button[kind="primary"] {
    color:white !important;
    background:linear-gradient(180deg,#4F6EFF,#3155D4) !important;
    border:1px solid #6281FF !important;
}
.stButton > button[kind="secondary"] {
    color:#D8E7FF !important;
    background:#0B1B34 !important;
    border:1px solid #2B5F9A !important;
}

.detected-title {
    color:#EAF2FF;
    font-size:.78rem;
    font-weight:850;
    margin-top:7px;
    margin-bottom:2px;
}
.detected-wait {
    color:#7E95B8;
    font-size:.68rem;
    margin-bottom:6px;
}
.mode-pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:6px 10px;
    border-radius:999px;
    background:#0A2C59;
    border:1px solid #2A75CC;
    color:#80BDFF;
    font-size:.72rem;
    font-weight:850;
}
.mode-dot {width:7px;height:7px;border-radius:99px;background:#1DE0A4;}

.demo-label {
    color:#AFC3E1;
    font-size:.70rem;
    font-weight:800;
    margin-top:8px;
    margin-bottom:4px;
}

.result-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    margin-bottom:8px;
}
.result-status-mini {
    display:inline-flex;
    align-items:center;
    height:26px;
    padding:0 9px;
    border-radius:999px;
    background:#0B2851;
    border:1px solid #1E5C9D;
    color:#80A7D8;
    font-size:.66rem;
    white-space:nowrap;
}

.result-box {
    height:222px;
    background:#08172B;
    border:1px solid #16416F;
    border-radius:11px;
    padding:12px;
    overflow:hidden;
}
.result-title {
    color:#F1F6FF;
    font-size:12px;
    font-weight:850;
    margin-bottom:8px;
}
.result-muted {
    color:#8AA1C2;
    font-size:.74rem;
    line-height:1.4;
}
.result-card-grid {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:7px;
    margin-bottom:8px;
}
.result-card {
    background:#0B1F39;
    border:1px solid #1B497B;
    border-radius:8px;
    padding:8px;
}
.result-k {font-size:.58rem;color:#7F96B9;font-weight:900;text-transform:uppercase;}
.result-v {font-size:.72rem;color:#F3F7FF;font-weight:800;margin-top:3px;overflow-wrap:anywhere;}
.question-box {
    background:#0B2851;
    border:1px solid #2B68AA;
    border-radius:9px;
    padding:10px;
    margin-bottom:8px;
}
.question-label {font-size:.59rem;color:#8EBBFF;font-weight:900;text-transform:uppercase;}
.question-text {font-size:.82rem;color:#F4F8FF;font-weight:800;margin-top:4px;}

.activity-card {
    background:#08172B;
    border:1px solid #16416F;
    border-left:4px solid #4D82FF;
    border-radius:10px;
    padding:7px 8px;
    margin-bottom:6px;
}
.activity-label {
    font-size:.62rem;
    font-weight:900;
    letter-spacing:.04em;
    color:#8EBBFF;
}
.activity-text {
    color:#D8E6FA;
    font-size:.70rem;
    line-height:1.3;
    margin-top:3px;
}
.tool-box {
    background:#08172B;
    border:1px solid #16416F;
    border-radius:10px;
    padding:9px;
    margin-top:8px;
}
.tool-title {
    color:#F2F7FF;
    font-size:.78rem;
    font-weight:850;
    margin-bottom:5px;
}
.tool-json {
    color:#A9CEF9;
    font-family:Consolas,monospace;
    font-size:.56rem;
    line-height:1.18;
    white-space:pre-wrap;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FIXED WORKFLOW UI
# ============================================================
def steps_for(mode: str):
    if mode == "REPLAY":
        return [
            ("Compile", "Build replay spec", "hypothesis_compiler.py", "#3E8DFF"),
            ("Set Time", "Lock historical clock", "replay-specific", "#45C79B"),
            ("As-Seen State", "Use data known then", "data_tools.py", "#7B64FF"),
            ("Evaluate", "Run quant checks", "quant tools", "#D9A52E"),
            ("Advance", "Move replay clock", "replay-specific", "#45B6D7"),
            ("Reason", "Interpret state", "LLM reasoning", "#CC56B7"),
            ("Outcome", "Ex-post compare", "agent_controller.py", "#438CFF"),
        ]
    if mode == "LIVE":
        return [
            ("Compile", "Build live spec", "hypothesis_compiler.py", "#3E8DFF"),
            ("Fetch Latest", "Newest market data", "live-specific", "#45C79B"),
            ("Build State", "Current market state", "data_tools.py", "#7B64FF"),
            ("Signal", "Run quant checks", "quant tools", "#D9A52E"),
            ("Reason", "Interpret evidence", "LLM reasoning", "#45B6D7"),
            ("Monitor", "Refresh / thresholds", "live-specific", "#CC56B7"),
            ("Output", "Current interpretation", "agent_controller.py", "#438CFF"),
        ]
    return [
        ("Compile", "Build research spec", "hypothesis_compiler.py", "#3E8DFF"),
        ("Validate", "Check feasibility", "compiler_validator.py", "#45C79B"),
        ("Load History", "Fetch historical data", "data_tools.py", "#7B64FF"),
        ("Quant Test", "Run statistical tests", "hypothesis_tests.py", "#D9A52E"),
        ("Reason", "Interpret evidence", "LLM reasoning", "#45B6D7"),
        ("Robustness", "Stability checks", "robustness.py", "#CC56B7"),
        ("Decision", "STOP / CONTINUE", "agent_controller.py", "#438CFF"),
    ]


def workflow_html(mode: str, states: list[tuple[str, str]]) -> str:
    cards = []
    for i, ((title, subtitle, fname, color), (state, caption)) in enumerate(zip(steps_for(mode), states), 1):
        border = {
            "done": "#35C48D",
            "active": color,
            "blocked": "#D8A133",
            "error": "#D65C6A",
            "todo": "#52627D",
        }.get(state, "#52627D")
        opacity = "1" if state in {"done", "active", "blocked", "error"} else ".72"
        cards.append(f"""
        <div class="card" style="border-color:{border};opacity:{opacity};">
            <div class="top">
                <span class="num">{i}</span>
                <span class="dot" style="background:{border};"></span>
            </div>
            <div class="title">{title}</div>
            <div class="sub">{caption}</div>
            <div class="file">{fname}</div>
        </div>
        """)
    return f"""
    <html><head><style>
    *{{box-sizing:border-box}}
    html,body{{margin:0;overflow:hidden;background:transparent;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Arial,sans-serif}}
    .grid{{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:6px;width:100%}}
    .card{{height:130px;background:linear-gradient(180deg,#0B213F,#08172B);
      border:1px solid;border-radius:10px;padding:8px;overflow:hidden}}
    .top{{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}}
    .num{{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
      justify-content:center;background:#1B2C49;color:white;font-size:10px;font-weight:850}}
    .dot{{width:6px;height:6px;border-radius:50%;box-shadow:0 0 9px currentColor}}
    .title{{color:#F5F8FF;font-size:11.4px;line-height:1.14;font-weight:850;min-height:25px}}
    .sub{{color:#91A7C8;font-size:8.6px;line-height:1.23;height:28px;margin-top:3px}}
    .file{{margin-top:6px;padding:4px 5px;border-radius:6px;background:#0D2B50;
      border:1px solid #1A4779;color:#A9C2E6;font-size:8px;white-space:nowrap;
      overflow:hidden;text-overflow:ellipsis}}
    </style></head><body><div class="grid">{''.join(cards)}</div></body></html>
    """


# ============================================================
# BACKEND BRIDGE — data only, does NOT control layout
# ============================================================
def get_controller() -> AgentController:
    if "controller" not in st.session_state:
        st.session_state.controller = AgentController()
    return st.session_state.controller


def reset_backend():
    st.session_state.controller = AgentController()
    st.session_state.route_mode = None
    st.session_state.route_reason = None
    st.session_state.result = None
    st.session_state.data_result = None
    st.session_state.error = None
    st.session_state.activity = []


def enum_value(v):
    return getattr(v, "value", v)


def current_mode_label():
    mode = st.session_state.get("route_mode")
    if mode == ExecutionMode.HISTORICAL_REPLAY:
        return "REPLAY"
    if mode == ExecutionMode.LIVE_FORECAST:
        return "LIVE"
    if mode == ExecutionMode.RESEARCH:
        return "RESEARCH"
    return "RESEARCH"


def resolve_data(result):
    if result is None:
        return None
    spec = result.specification
    if spec.status != SpecificationStatus.VALID or not result.validation.is_valid:
        return None

    cols = required_source_columns(spec)
    path = Path(os.getenv(
        "MARKET_DATA_CSV",
        str(PROJECT_ROOT / "data" / "agent_historical_master_with_a16.csv"),
    ))
    payload = {
        "required_columns": list(cols),
        "path": str(path),
        "file_exists": path.is_file(),
    }
    if path.is_file():
        frame = load_market_data(path, cols)
        payload["rows_loaded"] = int(len(frame))
        payload["loaded_columns"] = list(frame.columns)
        payload["status"] = "loaded"
    else:
        payload["status"] = "file_not_found"
    return payload


def run_backend(request_text: str):
    reset_backend()
    st.session_state.controller = AgentController()
    controller = st.session_state.controller

    route = route_request(request_text)
    st.session_state.route_mode = route.mode
    st.session_state.route_reason = route.reason

    st.session_state.activity = [
        ("SYSTEM", "Request received."),
        ("ROUTER", f"{route.mode.value}: {route.reason}"),
    ]

    if route.mode == ExecutionMode.RESEARCH:
        result = controller.process(request_text)
        st.session_state.result = result
        st.session_state.activity.append(
            ("COMPILER", f"Specification status: {result.specification.status.value}")
        )
        st.session_state.activity.append(
            ("VALIDATOR", "PASS" if result.validation.is_valid else "FAIL")
        )
        st.session_state.data_result = resolve_data(result)
        if controller.state.pending_clarifications:
            st.session_state.activity.append(
                ("AGENT", "Waiting for user clarification.")
            )
        elif st.session_state.data_result and st.session_state.data_result.get("status") == "loaded":
            st.session_state.activity.append(
                ("DATA", f"Loaded {st.session_state.data_result['rows_loaded']:,} rows.")
            )
        return

    # Real route result; dedicated Replay/Live backend is not implemented.
    st.session_state.result = None
    st.session_state.data_result = None
    st.session_state.activity.append(
        ("PIPELINE", "Dedicated backend pipeline is not implemented yet.")
    )


def answer_clarification(answer: str):
    controller = get_controller()
    result = controller.answer_clarification(answer)
    st.session_state.result = result
    st.session_state.data_result = resolve_data(result)
    st.session_state.activity.append(("USER", f"Clarification: {answer}"))
    st.session_state.activity.append(
        ("COMPILER", f"Resumed status: {result.specification.status.value}")
    )
    st.session_state.activity.append(
        ("VALIDATOR", "PASS" if result.validation.is_valid else "FAIL")
    )
    if st.session_state.data_result and st.session_state.data_result.get("status") == "loaded":
        st.session_state.activity.append(
            ("DATA", f"Loaded {st.session_state.data_result['rows_loaded']:,} rows.")
        )


def workflow_states():
    mode = current_mode_label()

    if not st.session_state.get("route_mode"):
        return [("todo", "Waiting")] * 7

    if mode in {"REPLAY", "LIVE"}:
        return [
            ("done", "Mode routed"),
            ("todo", "Not implemented"),
            ("todo", "Not implemented"),
            ("todo", "Not implemented"),
            ("todo", "Not implemented"),
            ("todo", "Not implemented"),
            ("todo", "Not implemented"),
        ]

    result = st.session_state.get("result")
    data_result = st.session_state.get("data_result")
    if result is None:
        return [("todo", "Waiting")] * 7

    spec = result.specification
    pending = bool(get_controller().state.pending_clarifications)

    s1 = ("done", spec.status.value)
    s2 = ("done" if result.validation.is_valid else "error",
          "PASS" if result.validation.is_valid else "FAIL")

    if pending:
        s3 = ("blocked", "Waiting for clarification")
    elif data_result and data_result.get("status") == "loaded":
        s3 = ("done", f"{data_result['rows_loaded']:,} rows")
    elif data_result and data_result.get("status") == "file_not_found":
        s3 = ("blocked", "CSV not found")
    else:
        s3 = ("todo", "Waiting")

    return [
        s1, s2, s3,
        ("todo", "Not implemented"),
        ("todo", "Not implemented"),
        ("todo", "Not implemented"),
        ("todo", "Not implemented"),
    ]


def compact_output():
    mode = st.session_state.get("route_mode")
    result = st.session_state.get("result")
    data = st.session_state.get("data_result")

    payload = {
        "mode": mode.value if mode else None,
        "status": result.specification.status.value if result else None,
        "validation": result.validation.is_valid if result else None,
        "pending_clarifications": len(get_controller().state.pending_clarifications),
    }
    if data:
        payload["data"] = {
            "status": data.get("status"),
            "rows_loaded": data.get("rows_loaded"),
            "required_columns": data.get("required_columns"),
        }
    return payload


# ============================================================
# SESSION DEFAULTS
# ============================================================
if "request_text" not in st.session_state:
    st.session_state.request_text = (
        "Do downward wind forecast revisions make ID1 more likely to exceed ID3 "
        "for evening delivery products?"
    )
if "controller" not in st.session_state:
    st.session_state.controller = AgentController()
if "activity" not in st.session_state:
    st.session_state.activity = []
if "result" not in st.session_state:
    st.session_state.result = None
if "route_mode" not in st.session_state:
    st.session_state.route_mode = None
if "route_reason" not in st.session_state:
    st.session_state.route_reason = None
if "data_result" not in st.session_state:
    st.session_state.data_result = None
if "error" not in st.session_state:
    st.session_state.error = None


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="pm-header">
  <div>
    <div class="pm-title">⚡ Power-Market Quant Research Agent</div>
    <div class="pm-subtitle">Natural-language research → automatic routing → quantitative tools → evidence-driven decisions</div>
  </div>
  <div class="pm-badge">Hackathon Demo UI</div>
</div>
""",
    unsafe_allow_html=True,
)

left, center, right = st.columns([2.3, 7.1, 2.75], gap="small")

# ============================================================
# LEFT — fixed
# ============================================================
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="panel-icon">▦</span>Research Request</div>', unsafe_allow_html=True)

        st.markdown(
            """
<div class="market-label">Market / Region</div>
<div class="market-pill"><span class="flag"></span><span>Germany (DE-LU)</span></div>
""",
            unsafe_allow_html=True,
        )

        pending_now = (
            st.session_state.route_mode == ExecutionMode.RESEARCH
            and bool(get_controller().state.pending_clarifications)
        )

        input_label = (
            "Clarification answer"
            if pending_now
            else "Natural-language request"
        )
        input_placeholder = (
            "Type your clarification answer here, e.g. Last 1 year"
            if pending_now
            else "Ask a research question, request a replay, or ask about the current market..."
        )

        st.text_area(
            input_label,
            key="request_text",
            placeholder=input_placeholder,
        )

        if pending_now:
            pending_question = get_controller().state.pending_clarifications[0].question
            st.markdown(
                f'<div class="clarify-hint"><b>Agent asks:</b> {pending_question}</div>',
                unsafe_allow_html=True,
            )

        if st.button("▶ Run Agent", type="primary", use_container_width=True):
            try:
                user_text = st.session_state.request_text.strip()

                if (
                    st.session_state.route_mode == ExecutionMode.RESEARCH
                    and get_controller().state.pending_clarifications
                ):
                    # Continue the existing Research conversation.
                    # The text box now acts as the clarification-answer input.
                    answer_clarification(user_text)
                else:
                    # Start a new top-level request.
                    run_backend(user_text)

                st.session_state.error = None
            except Exception as exc:
                st.session_state.error = f"{type(exc).__name__}: {exc}"
            st.rerun()

        if st.button("↻ Reset", use_container_width=True):
            request = st.session_state.request_text
            reset_backend()
            st.session_state.request_text = request
            st.rerun()

        st.markdown('<div class="detected-title">Detected mode (auto)</div>', unsafe_allow_html=True)
        if st.session_state.route_mode:
            st.markdown(
                f'<span class="mode-pill"><span class="mode-dot"></span>{current_mode_label()}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="detected-wait">Waiting for the router.</div>', unsafe_allow_html=True)

        st.markdown('<div class="demo-label">Demo examples</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns([1.2, 1.0, .75], gap="small")
        with d1:
            if st.button("Research", key="demo_research", use_container_width=True):
                st.session_state.request_text = (
                    "Do downward wind forecast revisions make ID1 more likely to exceed ID3 "
                    "for evening delivery products?"
                )
                st.rerun()
        with d2:
            if st.button("Replay", key="demo_replay", use_container_width=True):
                st.session_state.request_text = (
                    "Replay 2025-06-12 from 08:00 and show what the agent would have known."
                )
                st.rerun()
        with d3:
            if st.button("Live", key="demo_live", use_container_width=True):
                st.session_state.request_text = (
                    "What does the latest wind forecast revision imply for tonight's intraday market?"
                )
                st.rerun()


# ============================================================
# CENTER — fixed workflow + fixed results panel
# ============================================================
with center:
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="panel-icon">⌘</span>Agent Workflow</div>', unsafe_allow_html=True)
        components.html(
            workflow_html(current_mode_label(), workflow_states()),
            height=142,
            scrolling=False,
        )

    with st.container(border=True):
        mode_label = current_mode_label() if st.session_state.route_mode else "WAITING"
        st.markdown(
            f"""
<div class="result-head">
  <div class="panel-heading" style="margin:0;"><span class="panel-icon">▥</span>Research Results</div>
  <span class="result-status-mini">{mode_label}</span>
</div>
""",
            unsafe_allow_html=True,
        )

        result = st.session_state.result
        error = st.session_state.error

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        if error:
            st.error(error)

        elif not st.session_state.route_mode:
            st.markdown('<div class="result-title">Waiting for request</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="result-muted">Enter a natural-language request and click Run Agent. '
                'This panel will show the real backend result without changing the page layout.</div>',
                unsafe_allow_html=True,
            )

        elif current_mode_label() in {"REPLAY", "LIVE"}:
            st.markdown(
                f'<div class="result-title">{current_mode_label()} routed successfully</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="result-muted">The top-level mode router is real. '
                'This dedicated backend pipeline is not implemented yet, so no result is fabricated.</div>',
                unsafe_allow_html=True,
            )

        elif result is not None:
            spec = result.specification
            pending = get_controller().state.pending_clarifications

            cards = [
                ("Compiler", spec.status.value),
                ("Validation", "PASS" if result.validation.is_valid else "FAIL"),
                ("Output type", enum_value(spec.output_type) if spec.output_type else "—"),
                ("Event", f"{spec.event.variable} {spec.event.operator} {spec.event.value}" if spec.event else "—"),
            ]
            html = '<div class="result-card-grid">'
            for k, v in cards:
                html += (
                    f'<div class="result-card"><div class="result-k">{k}</div>'
                    f'<div class="result-v">{v}</div></div>'
                )
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            if pending:
                req = pending[0]
                st.markdown(
                    f"""
<div class="question-box">
  <div class="question-label">Agent asks for clarification</div>
  <div class="question-text">{req.question}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="result-muted">Type your answer in the main input box on the left, then click Run Agent.</div>',
                    unsafe_allow_html=True,
                )

            elif spec.status == SpecificationStatus.VALID:
                data = st.session_state.data_result
                if data and data.get("status") == "loaded":
                    st.markdown(
                        f'<div class="result-title">Real historical data loaded: {data["rows_loaded"]:,} rows</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="result-muted">Compiler → validator → required-column resolution → CSV load is complete. '
                        'Quantitative hypothesis testing is the next unfinished backend step.</div>',
                        unsafe_allow_html=True,
                    )
                elif data and data.get("status") == "file_not_found":
                    st.markdown(
                        '<div class="result-title">Specification is VALID</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="result-muted">Required data columns were resolved, but the configured CSV file was not found.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-title">Specification is VALID</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# RIGHT — fixed activity / tool output panels
# ============================================================
with right:
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="panel-icon">〽</span>Agent Activity</div>', unsafe_allow_html=True)

        if not st.session_state.activity:
            default_events = [
                ("SYSTEM", "Waiting for a natural-language request."),
                ("ROUTER", "Mode has not been detected yet."),
                ("AGENT", "No backend step has executed yet."),
                ("TOOL CALL", "No tool has been called yet."),
                ("TOOL OUTPUT", "No tool output yet."),
            ]
            events = default_events
        else:
            events = st.session_state.activity[-5:]

        for label, message in events:
            st.markdown(
                f"""
<div class="activity-card">
  <div class="activity-label">{label}</div>
  <div class="activity-text">{message}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
<div class="tool-box">
  <div class="tool-title">Latest Tool Output</div>
  <div class="tool-json">{json.dumps(compact_output(), indent=2, default=str)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
