import ast
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import app.agent_runtime as runtime_module
from app.agent_runtime import AgentRuntime
from app.mode_router import RouteDecision
from app.state import AgentResult
from domain.research_language import OutputType
from execution.research_pipeline import execute_research_specification
from schemas.hypothesis_schema import (
    Condition,
    ExecutionMode,
    ResearchSpecification,
    SpecificationStatus,
    StudyPeriod,
    StudyPeriodType,
    ValidationResult,
)
from tools.data_source import resolve_master_dataset_path
from tools.data_tools import required_source_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = PROJECT_ROOT / "app" / "agent_runtime.py"
UI_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"


def _specification(*, status=SpecificationStatus.VALID, with_period=False):
    return ResearchSpecification(
        original_hypothesis="Test total wind revision against ID1 minus ID3.",
        status=status,
        output_type=OutputType.PROBABILITY,
        event=Condition("id1_minus_id3", ">", 0),
        conditions=(Condition("total_wind_revision", "<", 0),),
        study_period=(
            StudyPeriod(type=StudyPeriodType.ALL_AVAILABLE)
            if with_period
            else None
        ),
    )


def _agent_result(*, status=SpecificationStatus.VALID, valid=True, with_period=False):
    return AgentResult(
        specification=_specification(status=status, with_period=with_period),
        validation=ValidationResult(
            is_valid=valid,
            errors=() if valid else ("invalid specification",),
        ),
    )


def _route() -> RouteDecision:
    return RouteDecision(ExecutionMode.RESEARCH, "test route")


def _write_master(path: Path, *, include_local=False) -> Path:
    data = {
        "delivery_start_utc": [
            "2026-01-01T11:00:00Z",
            "2026-01-01T11:15:00Z",
        ],
        "delivery_start_local": [
            "2026-01-01T12:00:00+01:00",
            "2026-01-01T12:15:00+01:00",
        ],
        "id1_price_eur_mwh": [30.0, 40.0],
        "id3_price_eur_mwh": [20.0, 25.0],
        "wind_onshore_a40": [100.0, 110.0],
        "wind_offshore_a40": [50.0, 55.0],
        "wind_onshore_a01": [90.0, 120.0],
        "wind_offshore_a01": [45.0, 50.0],
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _finalize(result: AgentResult):
    runtime = AgentRuntime.__new__(AgentRuntime)
    return runtime._finalize_research("request", _route(), result, [])


def _ui_resolve_data():
    source = UI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_data"
    )
    namespace = {
        "asdict": asdict,
        "SpecificationStatus": SpecificationStatus,
        "resolve_master_dataset_path": resolve_master_dataset_path,
        "execute_research_specification": execute_research_specification,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(UI_PATH), "exec"), namespace)
    return namespace["resolve_data"], namespace


def test_active_consumers_do_not_resolve_environment_or_hide_default_paths() -> None:
    for path in (RUNTIME_PATH, UI_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names)
            for node in tree.body
        )
        assert "os.getenv" not in source
        assert "os.environ" not in source
        assert "agent_historical_master_with_a16.csv" not in source
        assert "load_market_data" not in source


def test_runtime_executes_quantitative_research_through_shared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_master(tmp_path / "master.csv")
    monkeypatch.setenv("MARKET_DATA_CSV", str(csv_path))

    result = _finalize(_agent_result())

    assert result.status == "research_complete"
    assert result.data_summary["path"] == str(csv_path.resolve())
    assert result.data_summary["rows_loaded"] == 2
    assert result.data_summary["required_columns"] == list(
        required_source_columns(_specification())
    )
    assert "id1_minus_id3" in result.data_summary["columns_loaded"]
    assert "total_wind_revision" in result.data_summary["columns_loaded"]
    assert result.data_summary["quantitative_result"] is not None


def test_runtime_adds_local_time_canonical_variable_for_temporal_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_master(tmp_path / "master.csv", include_local=True)
    monkeypatch.setenv("MARKET_DATA_CSV", str(csv_path))

    result = _finalize(_agent_result(with_period=True))

    assert result.status == "research_complete"
    assert result.data_summary["required_columns"][-1] == "delivery_start_local"
    assert "delivery_start_local" in result.data_summary["columns_loaded"]


@pytest.mark.parametrize("configured", [None, "missing.csv"])
def test_runtime_preserves_non_loaded_flow_for_configuration_and_file_errors(
    configured: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if configured is None:
        monkeypatch.delenv("MARKET_DATA_CSV", raising=False)
    else:
        monkeypatch.setenv("MARKET_DATA_CSV", str(tmp_path / configured))

    result = _finalize(_agent_result())

    assert result.status == "data_not_configured"
    assert result.data_summary is None
    assert result.events[-1].label == "DATA"
    assert result.events[-1].status == "not_configured"


@pytest.mark.parametrize(
    "agent_result",
    [
        _agent_result(status=SpecificationStatus.NEEDS_CLARIFICATION),
        _agent_result(valid=False),
    ],
)
def test_runtime_does_not_load_for_clarification_or_invalid_specification(
    agent_result: AgentResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("data source must not be called")

    monkeypatch.setattr(runtime_module, "resolve_master_dataset_path", fail_if_called)
    monkeypatch.setattr(runtime_module, "execute_research_specification", fail_if_called)

    result = _finalize(agent_result)

    expected = (
        "needs_clarification"
        if agent_result.specification.status == SpecificationStatus.NEEDS_CLARIFICATION
        else "validation_error"
    )
    assert result.status == expected


def test_ui_resolve_data_exposes_real_quantitative_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_master(tmp_path / "master.csv")
    monkeypatch.setenv("MARKET_DATA_CSV", str(csv_path))
    resolve_data, _ = _ui_resolve_data()

    payload = resolve_data(_agent_result())

    assert payload["required_columns"] == list(required_source_columns(_specification()))
    assert payload["path"] == str(csv_path.resolve())
    assert payload["file_exists"] is True
    assert payload["rows_loaded"] == 2
    assert payload["status"] == "loaded"
    assert "id1_minus_id3" in payload["loaded_columns"]
    assert "total_wind_revision" in payload["loaded_columns"]
    assert payload["quantitative_result"] is not None


@pytest.mark.parametrize("configured", [None, "missing.csv"])
def test_ui_resolve_data_preserves_non_loaded_flow(
    configured: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if configured is None:
        monkeypatch.delenv("MARKET_DATA_CSV", raising=False)
    else:
        monkeypatch.setenv("MARKET_DATA_CSV", str(tmp_path / configured))
    resolve_data, _ = _ui_resolve_data()

    payload = resolve_data(_agent_result())

    assert payload["status"] == "file_not_found"
    assert payload["file_exists"] is False
    assert payload["path"] is None
    assert payload["error"]


@pytest.mark.parametrize(
    "result",
    [
        None,
        _agent_result(status=SpecificationStatus.NEEDS_CLARIFICATION),
        _agent_result(valid=False),
    ],
)
def test_ui_does_not_load_for_absent_clarification_or_invalid_result(result) -> None:
    resolve_data, namespace = _ui_resolve_data()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("data source must not be called")

    namespace["resolve_master_dataset_path"] = fail_if_called
    namespace["execute_research_specification"] = fail_if_called

    assert resolve_data(result) is None


def test_runtime_and_ui_delegate_to_research_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = _write_master(tmp_path / "master.csv")
    captured_runtime = []

    monkeypatch.setattr(runtime_module, "resolve_master_dataset_path", lambda: csv_path)

    quant = SimpleNamespace()
    pipeline_result = SimpleNamespace(
        prepared_data=pd.read_csv(csv_path),
        quantitative_result=None,
        required_source_columns=(),
        resolved_study_period=None,
        interpretation=None,
    )

    def runtime_executor(*args):
        captured_runtime.append(args)
        return pipeline_result

    monkeypatch.setattr(runtime_module, "execute_research_specification", runtime_executor)
    _finalize(_agent_result())

    resolve_data, namespace = _ui_resolve_data()
    captured_ui = []
    namespace["resolve_master_dataset_path"] = lambda: csv_path

    def ui_executor(*args):
        captured_ui.append(args)
        return pipeline_result

    namespace["execute_research_specification"] = ui_executor
    resolve_data(_agent_result())

    assert len(captured_runtime) == 1
    assert len(captured_ui) == 1
    assert captured_runtime[0][0] == _specification()
    assert captured_runtime[0][2] == csv_path
    assert captured_ui[0][0] == _specification()
    assert captured_ui[0][2] == csv_path
