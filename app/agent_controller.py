"""Orchestration for compilation, validation, and clarification state."""

from app.state import (
    AgentResult,
    AgentState,
    apply_compilation_result,
    record_clarification_answer,
)
from compiler.compiler_validator import validate_specification
from compiler.hypothesis_compiler import (
    compile_hypothesis,
    resume_hypothesis_compilation,
)


class AgentController:
    """Connect user input to compilation and validation."""

    def __init__(self) -> None:
        self.state = AgentState()

    def process(self, hypothesis: str) -> AgentResult:
        """Start a new top-level question and replace prior conversation state."""
        self.state = AgentState(original_question=hypothesis)
        specification = compile_hypothesis(hypothesis)
        validation = validate_specification(specification)
        self.state = apply_compilation_result(self.state, specification)
        return AgentResult(specification=specification, validation=validation)

    def answer_clarification(self, answer_text: str) -> AgentResult:
        """Continue the current question using a raw clarification answer."""
        if self.state.specification is None:
            raise ValueError(
                "Cannot answer clarification before processing an initial question."
            )

        recorded_state = record_clarification_answer(self.state, answer_text)
        resumed_specification = resume_hypothesis_compilation(
            previous_specification=recorded_state.specification,
            pending_clarifications=recorded_state.pending_clarifications,
            answer_text=answer_text,
        )
        validation = validate_specification(resumed_specification)
        self.state = apply_compilation_result(
            recorded_state,
            resumed_specification,
        )
        return AgentResult(
            specification=resumed_specification,
            validation=validation,
        )
