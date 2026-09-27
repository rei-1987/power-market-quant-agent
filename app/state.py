"""Immutable state containers shared by the application layer."""

from dataclasses import asdict, dataclass, replace
import json

from schemas.hypothesis_schema import (
    ClarificationRequest,
    ResearchSpecification,
    SpecificationStatus,
    ValidationResult,
)


@dataclass(frozen=True)
class ClarificationTurn:
    """One raw user answer and the clarification requests pending at that time."""

    requests: tuple[ClarificationRequest, ...]
    answer_text: str


@dataclass(frozen=True)
class AgentState:
    """Conversation memory for compiling and clarifying one original question."""

    original_question: str | None = None
    specification: ResearchSpecification | None = None
    pending_clarifications: tuple[ClarificationRequest, ...] = ()
    clarification_history: tuple[ClarificationTurn, ...] = ()


def apply_compilation_result(
    state: AgentState,
    specification: ResearchSpecification,
) -> AgentState:
    """Store a compiler result and replace pending questions from that result."""
    original_question = state.original_question
    if original_question is None:
        original_question = specification.original_hypothesis
    return replace(
        state,
        original_question=original_question,
        specification=specification,
        pending_clarifications=specification.clarification_requests,
    )


def record_clarification_answer(
    state: AgentState,
    answer_text: str,
) -> AgentState:
    """Append a raw answer while retaining the currently pending questions."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("Clarification answer must be a non-empty string.")
    if not state.pending_clarifications:
        raise ValueError(
            "Cannot record a clarification answer when no clarification is pending."
        )
    turn = ClarificationTurn(
        requests=state.pending_clarifications,
        answer_text=answer_text,
    )
    return replace(
        state,
        clarification_history=(*state.clarification_history, turn),
    )


def needs_clarification(state: AgentState) -> bool:
    """Return whether the current conversation is awaiting clarification."""
    return bool(state.pending_clarifications) or (
        state.specification is not None
        and state.specification.status == SpecificationStatus.NEEDS_CLARIFICATION
    )


def pending_clarification_fields(state: AgentState) -> tuple[str, ...]:
    """Return pending specification fields in stable order without duplicates."""
    return tuple(
        dict.fromkeys(request.field for request in state.pending_clarifications)
    )


@dataclass(frozen=True)
class AgentResult:
    specification: ResearchSpecification
    validation: ValidationResult

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
