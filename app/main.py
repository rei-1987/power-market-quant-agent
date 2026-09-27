"""Command-line entry point for the mock research agent."""

from app.agent_controller import AgentController


def main() -> None:
    hypothesis = input("Enter a power-market hypothesis: ").strip()
    result = AgentController().process(hypothesis)
    print(result.to_json())


if __name__ == "__main__":
    main()
