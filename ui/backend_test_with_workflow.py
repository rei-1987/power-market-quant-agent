"""Backend test UI that preserves the original 7-step Agent Workflow framework.

This page uses the *real current backend*:
- AgentController
- compiler
- validator
- clarification state
- required source-column resolution
- real CSV loading when configured

No fake statistics, fake backtest, or fake research result is shown.
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
from schemas.hypothesis_schema import SpecificationStatus
from tools.data_tools import required_source_columns, load_market_data

st.set_page_config(
    page_title="Power-Market Quant Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
[data-testid="stHeader"], [data-testid="stToolbar"] {display:none !important;}
#MainMenu, footer {visibility:hidden !important;}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:#071426 !important;
    color:#EAF2FF !important;
}
.block-container {max-width:1700px;padding-top:.8rem;padding-bottom:1rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(180deg,#0B1B34,#09192F) !important;
    border:1px solid #17467C !important;
    border-radius:13px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {padding:.72rem .78rem !important;}
.pm-header {
    background:linear-gradient(180deg,#0B1A31,#08162A);
    border:1px solid #173C6B;border-radius:14px;padding:13px 18px;margin-bottom:9px;
}
.pm-title {font-size:1.7rem;font-weight:850;color:#F4F8FF;}
.pm-subtitle {font-size:.80rem;color:#91A7C8;margin-top:5px;}
.panel-title {font-size:.98rem;font-weight:850;color:#F2F7FF;margin-bottom:.42rem;}
.small {color:#7F96B9;font-size:.72rem;line-height:1.35;}
.mode-pill {
    display:inline-block;padding:5px 9px;border-radius:999px;
    background:#0A2C59;border:1px solid #2A75CC;color:#80BDFF;font-size:.68rem;font-weight:850;
}
.question-box {
    background:#0A2447;border:1px solid #2D6BAA;border-radius:10px;padding:11px 12px;
}
.question-title {font-size:.64rem;color:#8EBBFF;font-weight:900;text-transform:uppercase;}
.question-text {font-size:.89rem;color:#F6FAFF;font-weight:800;margin-top:4px;}
.card-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;}
.card {
    background:#08172B;border:1px solid #16416F;border-radius:9px;padding:9px;min-height:68px;
}
.card-label {font-size:.59rem;color:#8099BD;font-weight:900;text-transform:uppercase;}
.card-value {font-size:.78rem;color:#F4F8FF;font-weight:800;margin-top:4px;overflow-wrap:anywhere;}
div[data-testid="stTextArea"] textarea {
    background:#08172B !important;color:#EAF2FF !important;border:1px solid #2A5D97 !important;
}
.stButton > button {border-radius:9px;font-weight:800;}
[data-testid="stExpander"] {
    border:1px solid #173F70 !important;border-radius:9px !important;background:#08172B !important;
}
div[data-testid="stCodeBlock"] pre {font-size:.62rem !important;line-height:1.2 !important;}
</style>
""",
    unsafe_allow_html=True,
)


def get_controller() -> AgentController:
    if "wf_controller" not in st.session_state:
        st.session_state.wf_controller = AgentController()
    return st.session_state.wf_controller


def reset_all() -> None:
    st.session_state.wf_controller = AgentController()
    st.session_state.wf_result = None
    st.session_state.wf_error = None
    st.session_state.wf_data = None


def value_of(v):
    return getattr(v, "value", v)


def condition_text(c):
    if c is None:
        return "—"
    return f"{c.variable} {c.operator} {c.value}"


def study_text(p):
    if p is None:
        return "Not set"
    t = value_of(p.type)
    if t == "relative_lookback":
        return f"Last {p.lookback_value} {value_of(p.lookback_unit)}"
    if t == "absolute_date_range":
        return f"{p.start_date} → {p.end_date}"
    if t == "all_available":
        return "All available history"
    return str(p)


def data_boundary(result):
    spec = result.specification
    validation = result.validation
    if spec.status != SpecificationStatus.VALID or not validation.is_valid:
        return None

    columns = required_source_columns(spec)
    configured = os.getenv(
        "MARKET_DATA_CSV",
        str(PROJECT_ROOT / "data" / "agent_historical_master_with_a16.csv"),
    )
    path = Path(configured)

    payload = {
        "required_columns": list(columns),
        "csv_path": str(path),
        "file_exists": path.is_file(),
    }

    if path.is_file():
        frame = load_market_data(path, columns)
        payload["rows_loaded"] = int(len(frame))
        payload["loaded_columns"] = list(frame.columns)
        payload["status"] = "loaded"
    else:
        payload["status"] = "file_not_found"
    return payload


def workflow_html(result, data_result):
    if result is None:
        states = ["waiting"] * 7
        captions = [
            "Waiting", "Waiting", "Waiting", "Not implemented",
            "Not implemented", "Not implemented", "Not implemented",
        ]
    else:
        spec = result.specification
        validation = result.validation
        has_pending = bool(spec.clarification_requests)

        compile_state = "done" if spec is not None else "waiting"
        validate_state = "done" if validation.is_valid else "error"

        if has_pending:
            load_state = "blocked"
        elif spec.status == SpecificationStatus.VALID:
            if data_result and data_result.get("status") == "loaded":
                load_state = "done"
            elif data_result:
                load_state = "blocked"
            else:
                load_state = "waiting"
        else:
            load_state = "blocked"

        states = [
            compile_state,
            validate_state,
            load_state,
            "todo",
            "todo",
            "todo",
            "todo",
        ]
        captions = [
            value_of(spec.status),
            "PASS" if validation.is_valid else "FAIL",
            (
                f"{data_result.get('rows_loaded', 0):,} rows"
                if data_result and data_result.get("status") == "loaded"
                else ("Blocked by clarification" if has_pending else "Waiting")
            ),
            "Not implemented",
            "Not implemented",
            "Not implemented",
            "Not implemented",
        ]

    steps = [
        ("Compile Request", "hypothesis_compiler.py"),
        ("Validate", "compiler_validator.py"),
        ("Load History", "data_tools.py"),
        ("Quant Test", "hypothesis_tests.py"),
        ("Reason", "LLM reasoning"),
        ("Robustness", "robustness.py"),
        ("Decision", "agent_controller.py"),
    ]
    colors = {
        "done": "#35C48D",
        "waiting": "#4A7DD8",
        "blocked": "#D8A133",
        "error": "#D65C6A",
        "todo": "#6F7891",
    }

    cards = []
    for i, ((title, filename), state, caption) in enumerate(zip(steps, states, captions), 1):
        color = colors[state]
        cards.append(
            f"""
            <div class="step" style="border-color:{color};">
              <div class="top"><span class="num">{i}</span><span class="dot" style="background:{color};"></span></div>
              <div class="title">{title}</div>
              <div class="caption">{caption}</div>
              <div class="file">{filename}</div>
            </div>
            """
        )

    return f"""
    <html><head><style>
    *{{box-sizing:border-box}}
    html,body{{margin:0;background:transparent;overflow:hidden;font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}}
    .grid{{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:7px;width:100%;}}
    .step{{height:145px;background:linear-gradient(180deg,#0B213F,#08172B);border:1px solid;border-radius:10px;padding:9px;}}
    .top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}}
    .num{{width:23px;height:23px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#1B2C49;color:white;font-size:10px;font-weight:900;}}
    .dot{{width:7px;height:7px;border-radius:50%;box-shadow:0 0 10px currentColor;}}
    .title{{font-size:12px;font-weight:850;color:#F5F8FF;line-height:1.18;min-height:30px;}}
    .caption{{font-size:9.5px;color:#9AB1D2;line-height:1.25;margin-top:5px;min-height:28px;}}
    .file{{font-size:8px;color:#A9C2E6;background:#0D2B50;border:1px solid #1A4779;border-radius:6px;padding:5px;margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
    </style></head><body><div class="grid">{''.join(cards)}</div></body></html>
    """


controller = get_controller()
result = st.session_state.get("wf_result")
data_result = st.session_state.get("wf_data")

st.markdown(
    """
<div class="pm-header">
  <div class="pm-title">⚡ Power-Market Quant Research Agent</div>
  <div class="pm-subtitle">Real backend test with the original Agent Workflow framework preserved</div>
</div>
""",
    unsafe_allow_html=True,
)

left, center, right = st.columns([2.25, 7.1, 2.65], gap="small")

# LEFT
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Research Request</div>', unsafe_allow_html=True)
        default_request = (
            "Do downward wind forecast revisions make ID1 more likely to exceed ID3 "
            "for evening delivery products?"
        )
        request = st.text_area(
            "Natural-language request",
            value=st.session_state.get("wf_request", default_request),
            height=150,
        )
        st.session_state.wf_request = request

        if st.button("Run backend", type="primary", use_container_width=True):
            reset_all()
            controller = get_controller()
            try:
                res = controller.process(request)
                st.session_state.wf_result = res
                st.session_state.wf_data = data_boundary(res)
            except Exception as exc:
                st.session_state.wf_error = f"{type(exc).__name__}: {exc}"
            st.rerun()

        if st.button("Reset", use_container_width=True):
            reset_all()
            st.rerun()

        st.markdown("**Current backend state**")
        if result is None:
            st.markdown('<span class="mode-pill">WAITING</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<span class="mode-pill">{value_of(result.specification.status).upper()}</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Validation: {result.validation.is_valid} · "
                f"Pending clarifications: {len(controller.state.pending_clarifications)}"
            )

# CENTER
with center:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Agent Workflow</div>', unsafe_allow_html=True)
        components.html(workflow_html(result, data_result), height=160, scrolling=False)

    with st.container(border=True):
        st.markdown('<div class="panel-title">Research State</div>', unsafe_allow_html=True)

        error = st.session_state.get("wf_error")
        if error:
            st.error(error)

        if result is None:
            st.markdown('<div class="small">Run the backend to populate the real research state.</div>', unsafe_allow_html=True)
        else:
            spec = result.specification
            cards = [
                ("Status", value_of(spec.status)),
                ("Output type", value_of(spec.output_type) if spec.output_type else "—"),
                ("Outcome event", condition_text(spec.event)),
                ("Study period", study_text(spec.study_period)),
            ]
            html = '<div class="card-grid">'
            for label, val in cards:
                html += f"""
                <div class="card">
                  <div class="card-label">{label}</div>
                  <div class="card-value">{val}</div>
                </div>
                """
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

            pending = controller.state.pending_clarifications
            if pending:
                st.write("")
                for i, req in enumerate(pending):
                    st.markdown(
                        f"""
<div class="question-box">
  <div class="question-title">Backend asks</div>
  <div class="question-text">{req.question}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    if req.options:
                        cols = st.columns(min(len(req.options), 5))
                        for j, option in enumerate(req.options):
                            with cols[j % len(cols)]:
                                if st.button(option.label, key=f"opt_{i}_{j}", use_container_width=True):
                                    try:
                                        resumed = controller.answer_clarification(option.label)
                                        st.session_state.wf_result = resumed
                                        st.session_state.wf_data = data_boundary(resumed)
                                        st.session_state.wf_error = None
                                    except Exception as exc:
                                        st.session_state.wf_error = f"{type(exc).__name__}: {exc}"
                                    st.rerun()

                    if req.allow_free_text:
                        answer = st.text_input("Or answer in free text", key=f"free_{i}")
                        if st.button("Submit answer", key=f"submit_{i}"):
                            try:
                                resumed = controller.answer_clarification(answer)
                                st.session_state.wf_result = resumed
                                st.session_state.wf_data = data_boundary(resumed)
                                st.session_state.wf_error = None
                            except Exception as exc:
                                st.session_state.wf_error = f"{type(exc).__name__}: {exc}"
                            st.rerun()

            elif spec.status == SpecificationStatus.VALID:
                st.success("Specification is VALID and clarification is complete.")

                if data_result:
                    st.write("")
                    st.markdown("**Real data boundary**")
                    if data_result.get("status") == "loaded":
                        st.success(f"Loaded {data_result['rows_loaded']:,} rows from the real CSV.")
                        st.markdown("Required columns:")
                        st.code("\n".join(data_result["required_columns"]))
                        st.warning(
                            "Next backend boundary: quantitative hypothesis testing "
                            "(tools/hypothesis_tests.py is not implemented yet)."
                        )
                    else:
                        st.warning("Required columns were resolved, but the configured CSV was not found.")
                        st.code("\n".join(data_result.get("required_columns", [])))

# RIGHT
with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Backend Interpretation</div>', unsafe_allow_html=True)
        if result is None:
            st.markdown('<div class="small">Waiting for a request.</div>', unsafe_allow_html=True)
        else:
            spec = result.specification
            if spec.status == SpecificationStatus.NEEDS_CLARIFICATION:
                st.markdown(
                    '<div class="small">The compiler understood part of the request, '
                    'but will not proceed until missing research information is supplied.</div>',
                    unsafe_allow_html=True,
                )
            elif spec.status == SpecificationStatus.VALID:
                st.markdown(
                    '<div class="small">The specification is executable at the current '
                    'compiler/validator boundary.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="small">The request is outside the supported research boundary.</div>',
                    unsafe_allow_html=True,
                )

    with st.expander("Raw backend JSON (debug only)", expanded=False):
        if result is None:
            st.code('{"status":"waiting"}', language="json")
        else:
            st.code(result.to_json(), language="json")
