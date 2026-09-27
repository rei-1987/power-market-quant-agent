# Frontend ↔ Backend Integration v1

This patch adds new files only:

- `app/mode_router.py`
  - Real OpenAI LLM routing to Research / Historical Replay / Live Forecast.
  - No keyword mock/fallback.
- `app/agent_runtime.py`
  - Connects router → existing `AgentController` → compiler → validator → real CSV data loader.
  - Does not invent statistical results.
- `ui/streamlit_app.py`
  - Calls the real runtime.
  - Displays real routing, compilation, validation, clarifications, and data-load status.
- `requirements_runtime.txt`

## Install

From the repo root:

```powershell
python -m pip install -r requirements_runtime.txt
```

## API key

If your `OPENAI_API_KEY` is already set in Windows, no change is needed.

Otherwise create a `.env` in the repo root:

```text
OPENAI_API_KEY=your_key_here
OPENAI_ROUTER_MODEL=gpt-5.6-luna
OPENAI_COMPILER_MODEL=gpt-5.6-luna
```

Do not commit `.env`.

## Market-data path

Research, Historical Replay, and Live Forecast share one master market dataset.
The CSV contains market history through the latest available observation. Set:

```text
MARKET_DATA_CSV=G:\path\to\agent_historical_master_with_a16.csv
```

in `.env`, or supply an explicit path override when calling the shared Data Core
loader. The shared loader has no hidden default path. Relative configured paths
are interpreted from the current process working directory.

For the current MVP, the future Live pipeline may interpret the latest available
observation in this dataset as simulated current time. That interpretation
belongs to the Live pipeline; the Shared Data Core exposes only the neutral
latest-available timestamp.

## Run

```powershell
python -m streamlit run ui/streamlit_app.py
```

## What is real in this patch?

- LLM mode routing
- Existing Research compiler
- Existing validator
- Clarification state
- Existing data dependency resolution
- Existing CSV column loading

## What is still not implemented?

- `tools/hypothesis_tests.py` currently raises `NotImplementedError`
- Historical Replay dedicated pipeline
- Live Forecast dedicated pipeline
- Robustness / backtest / reasoning loop beyond the currently implemented backend
