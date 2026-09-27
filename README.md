# Power Market Quant Research Agent

Minimal hackathon scaffold for turning a natural-language power-market hypothesis
into a validated mock research specification. It is deterministic, makes no API
calls, and does not require an OpenAI API key.

## Run the mock workflow

Requires Python 3.10 or newer.

```powershell
python -m app.main
```

Example input:

```text
Higher wind lowers day-ahead price
```

The command prints a JSON research specification and its validation result.

## Run tests

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

## Current scope

- `compiler/` contains the mock compiler, validator, and variable registry.
- `schemas/` defines the minimal data contract.
- `app/` connects compilation and validation through a small controller and CLI.
- `tools/` contains explicit placeholders for later quantitative components.

Real data access, quantitative tests, robustness analysis, backtesting, model tool
selection, and OpenAI API integration are intentionally out of scope for now.
