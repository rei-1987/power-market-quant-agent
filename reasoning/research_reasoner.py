"""Placeholder interface for the future LLM-based research reasoning layer.

The reasoning layer will eventually read a validated research specification and
the current evidence/state, recommend the next research action, and support
STOP, CONTINUE, or INCONCLUSIVE outcomes. No reasoning logic or OpenAI API
integration is implemented here yet.
"""


def recommend_next_action(
    validated_specification: object,
    current_evidence: object,
) -> None:
    """Reserve the interface for future research-action recommendations."""
    raise NotImplementedError("The research reasoning layer is not implemented yet.")
