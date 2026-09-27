"""Temporary frontend for testing the currently working Research backend loop.

This page directly exercises:
natural language -> AgentController -> compiler -> validator
-> clarification -> resumed compiler -> validator
-> required source columns -> real CSV load (if configured/present)

It intentionally bypasses the top-level Research/Replay/Live router so the
existing Research backend can be inspected by itself.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_controller import AgentController
from schemas.hypothesis_schema import SpecificationStatus
from tools.data_tools import required_source_columns, load_market_data


st.set_page_config(
    page_title="Backend Research Flow Test",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
<style>
[data-testid="stHeader"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
.stApp {background:#071426;color:#eef5ff;}
.block-container {max-width:1600px;padding-top:1rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#0b1b34 !important;
    border:1px solid #1c4778 !important;
    border-radius:14px !important;
}
h1,h2,h3,h4,p,label {color:#eef5ff !important;}
.small {color:#8fa6c7;font-size:.82rem;}
.pill {
    display:inline-block;
    padding:5px 10px;
    border-radius:999px;
    border:1px solid #2a6ab3;
    background:#0a2a55;
    color:#83bdff;
    font-weight:800;
    font-size:.78rem;
}
.ok {
    color:#75e0ad;
    font-weight:800;
}
.warn {
    color:#ffd166;
    font-weight:800;
}
.step {
    background:#08172b;
    border:1px solid #16416f;
    border-radius:10px;
    padding:10px 12px;
    min-height:78px;
}
.step-title {
    color:#8ebcff;
    font-size:.70rem;
    font-weight:900;
    text-transform:uppercase;
}
.step-value {
    color:#f2f7ff;
    font-size:.90rem;
    font-weight:800;
    margin-top:5px;
}
div[data-testid="stTextArea"] textarea {
    background:#08172b !important;
    color:#eef5ff !important;
    border:1px solid #2c5d96 !important;
}
.stButton > button {
    border-radius:9px;
    font-weight:800;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_controller() -> AgentController:
    if "backend_test_controller" not in st.session_state:
        st.session_state.backend_test_controller = AgentController()
    return st.session_state.backend_test_controller


def reset_state() -> None:
    st.session_state.backend_test_controller = AgentController()
    st.session_state.backend_test_result = None
    st.session_state.backend_test_error = None
    st.session_state.backend_test_data = None


def enum_value(value):
    return getattr(value, "value", value)


def stringify(value):
    if value is None:
        return "—"
    if is_dataclass(value):
        return json.dumps(asdict(value), default=str)
    if isinstance(value, tuple):
        if not value:
            return "—"
        return ", ".join(str(v) for v in value)
    return str(enum_value(value))


def render_spec(result):
    spec = result.specification
    validation = result.validation

    cols = st.columns(4)
    cards = [
        ("Compiler status", enum_value(spec.status)),
        ("Validation", "PASS" if validation.is_valid else "FAIL"),
        ("Target", spec.target or "—"),
        ("Output type", enum_value(spec.output_type) if spec.output_type else "—"),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
<div class="step">
  <div class="step-title">{label}</div>
  <div class="step-value">{value}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.write("")
    with st.expander("Compiled specification", expanded=True):
        st.json(asdict(spec))

    if validation.errors:
        for error in validation.errors:
            st.error(error)


def try_load_real_data(result):
    spec = result.specification
    validation = result.validation

    if spec.status != SpecificationStatus.VALID or not validation.is_valid:
        return None

    try:
        columns = required_source_columns(spec)
    except Exception as exc:
        return {
            "status": "dependency_resolution_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    configured = os.getenv(
        "MARKET_DATA_CSV",
        str(PROJECT_ROOT / "data" / "agent_historical_master_with_a16.csv"),
    )
    path = Path(configured)

    payload = {
        "status": "columns_resolved",
        "required_columns": list(columns),
        "csv_path": str(path),
        "file_exists": path.is_file(),
    }

    if not path.is_file():
        return payload

    try:
        frame = load_market_data(path, columns)
        payload.update(
            {
                "status": "data_loaded",
                "rows_loaded": int(len(frame)),
                "loaded_columns": list(frame.columns),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "data_load_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return payload


controller = get_controller()

st.title("⚡ Research Backend — Frontend Test")
st.markdown(
    '<div class="small">This page shows only what the current backend really does. '
    'No mock router, fake statistics, fake backtest, or fake chart.</div>',
    unsafe_allow_html=True,
)

left, main, right = st.columns([2.4, 6.4, 2.8], gap="medium")

with left:
    with st.container(border=True):
        st.subheader("1. Request")

        default_request = (
            "Do downward wind forecast revisions make ID1 more likely to exceed ID3 "
            "for evening delivery products?"
        )
        request = st.text_area(
            "Natural-language research request",
            value=st.session_state.get("backend_test_request", default_request),
            height=160,
        )
        st.session_state.backend_test_request = request

        if st.button("Run current backend", type="primary", use_container_width=True):
            reset_state()
            controller = get_controller()
            try:
                result = controller.process(request)
                st.session_state.backend_test_result = result
                st.session_state.backend_test_data = try_load_real_data(result)
            except Exception as exc:
                st.session_state.backend_test_error = f"{type(exc).__name__}: {exc}"
            st.rerun()

        if st.button("Reset", use_container_width=True):
            reset_state()
            st.rerun()

        st.markdown("---")
        st.markdown("**What this page tests**")
        st.markdown(
            """
- Compiler
- Validator
- Clarification state
- Clarification resume
- Required CSV columns
- Real CSV loading
"""
        )

with main:
    with st.container(border=True):
        st.subheader("2. Backend Flow")

        result = st.session_state.get("backend_test_result")
        error = st.session_state.get("backend_test_error")

        if error:
            st.error(error)

        elif result is None:
            st.info("Click **Run current backend** to start.")

        else:
            render_spec(result)

            pending = controller.state.pending_clarifications
            if pending:
                st.write("")
                st.subheader("Backend asks for clarification")

                for i, req in enumerate(pending):
                    st.markdown(f"**{req.question}**")

                    if req.options:
                        option_cols = st.columns(min(len(req.options), 5))
                        for j, option in enumerate(req.options):
                            with option_cols[j % len(option_cols)]:
                                if st.button(
                                    option.label,
                                    key=f"clarify_{i}_{j}",
                                    use_container_width=True,
                                ):
                                    try:
                                        resumed = controller.answer_clarification(option.label)
                                        st.session_state.backend_test_result = resumed
                                        st.session_state.backend_test_data = try_load_real_data(resumed)
                                        st.session_state.backend_test_error = None
                                    except Exception as exc:
                                        st.session_state.backend_test_error = (
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                    st.rerun()

                    if req.allow_free_text:
                        answer = st.text_input(
                            "Or answer in free text",
                            key=f"free_text_{i}",
                        )
                        if st.button(
                            "Submit clarification",
                            key=f"submit_free_{i}",
                        ):
                            try:
                                resumed = controller.answer_clarification(answer)
                                st.session_state.backend_test_result = resumed
                                st.session_state.backend_test_data = try_load_real_data(resumed)
                                st.session_state.backend_test_error = None
                            except Exception as exc:
                                st.session_state.backend_test_error = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                            st.rerun()

            else:
                st.success("No clarification is pending.")

            data_result = st.session_state.get("backend_test_data")
            if data_result:
                st.write("")
                st.subheader("3. Data boundary")
                st.json(data_result)

                if data_result.get("status") == "data_loaded":
                    st.success(
                        f"Real CSV load succeeded: {data_result['rows_loaded']:,} rows."
                    )
                    st.info(
                        "The next backend boundary is hypothesis testing. "
                        "tools/hypothesis_tests.py is not implemented yet."
                    )
                elif data_result.get("status") == "columns_resolved":
                    st.warning(
                        "The backend successfully resolved the required columns, "
                        "but the configured CSV file was not found."
                    )

with right:
    with st.container(border=True):
        st.subheader("4. Current State")

        result = st.session_state.get("backend_test_result")
        if result is None:
            st.markdown('<span class="pill">WAITING</span>', unsafe_allow_html=True)
        else:
            status = enum_value(result.specification.status)
            st.markdown(
                f'<span class="pill">{status.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(f"**Validation:** {result.validation.is_valid}")
            st.markdown(
                f"**Pending clarifications:** "
                f"{len(controller.state.pending_clarifications)}"
            )
            st.markdown(
                f"**Clarification turns:** "
                f"{len(controller.state.clarification_history)}"
            )

    with st.container(border=True):
        st.subheader("Raw backend output")
        result = st.session_state.get("backend_test_result")
        if result is None:
            st.code('{"status": "waiting"}', language="json")
        else:
            st.code(result.to_json(), language="json")
