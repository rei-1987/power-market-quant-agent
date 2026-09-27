# Power Market Quant Research Agent

A domain-specific AI agent for electricity-market quantitative research.

The agent takes natural-language power-market questions and automatically routes them into one of three execution modes:

- **Research Mode** — tests historical market hypotheses.
- **Historical Replay Mode** — reconstructs what the agent could have concluded at a past decision time using only information available then.
- **Live Forecast Mode** — evaluates the latest market data and produces a current market signal.

The current prototype focuses on the German **DE-LU power market**.

---

## Core idea

The system separates:

1. Natural-language understanding
2. Structured research specification
3. Data retrieval
4. Quantitative analysis
5. LLM reasoning
6. Research decision

The research workflow can return explicit outcomes such as:

- `CONTINUE`
- `STOP`
- `INCONCLUSIVE`

This allows the agent to stop weak hypotheses early instead of always producing a narrative answer.

---

## Run the demo

Requires **Python 3.10+**.

Install runtime dependencies:

```powershell
python -m pip install -r requirements_runtime.txt
```

Launch the Streamlit interface:

```powershell
streamlit run ui/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

The interface accepts natural-language questions and automatically routes each request to Research, Historical Replay, or Live Forecast.

The three buttons in the interface are only demo shortcuts. They prefill example requests while the router still determines the execution mode automatically.

---

## Example requests

### Research

```text
Do downward wind revisions increase the probability that ID1 > ID3?
```

### Historical Replay

```text
What would the agent have concluded at 08:00 on 2025-06-12?
```

### Live Forecast

```text
What does the latest wind forecast revision imply for tonight's intraday market?
```

---

## Architecture

Main components:

- `routing/` — request understanding and execution-mode selection
- `compiler/` — hypothesis compilation, validation, and variable registry
- `execution/` — research, historical replay, and live pipelines
- `tools/` — data access and quantitative analysis tools
- `reasoning/` — interpretation of quantitative evidence and research decisions
- `schemas/` — structured research specification
- `ui/` — Streamlit demo interface
- `tests/` — automated tests

A detailed architecture diagram is included in:

`agent framework 2026_09_27_13_40.png`

---

## Environment variables

Copy `.env.example` to `.env`.

Available configuration:

```text
OPENAI_API_KEY=
MARKET_DATA_CSV=
```

Do not commit `.env`.

The repository already excludes `.env` through `.gitignore`.

---

## Market data

The agent is designed around DE-LU power-market data, including variables such as:

- day-ahead prices
- intraday prices
- wind forecasts
- solar forecasts
- generation
- load
- forecast revisions

Historical and live execution use the configured market dataset through `MARKET_DATA_CSV`.

Private or licensed market datasets are not included in this repository.

---

## Run tests

Install development dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the test suite:

```powershell
python -m pytest
```

---

## Design principles

### Point-in-time correctness

Historical replay only uses information available at the requested decision time, helping avoid future-information leakage.

### Structured research specification

Natural-language questions are converted into structured research specifications before execution.

### Separation of evidence and reasoning

Quantitative calculations are performed by deterministic tools.

The reasoning layer interprets the evidence and determines the next research action rather than inventing numerical results.

### Explicit research decisions

The agent can return `CONTINUE`, `STOP`, or `INCONCLUSIVE` depending on the available evidence.

---

## Hackathon scope

This repository was developed for the **X-IA Hackathon — Rise of Agents X**.

The current prototype demonstrates the core architecture and end-to-end execution logic of a power-market quantitative research agent across:

- historical research
- historical replay
- live market analysis

This is a research prototype and not a production trading system.
