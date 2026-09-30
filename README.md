# Power Market Quant Research Agent

A domain-specific AI framework for electricity-market quantitative research.

The system takes natural-language power-market questions, converts them into structured research tasks, applies point-in-time data controls, executes quantitative analysis, and returns evidence-based research conclusions.

The current implementation focuses on the **German DE-LU power market**, while the architecture is designed to separate reusable agent and quantitative logic from market-specific data, features, products, and rules.

---

## What the agent does

Users can ask power-market questions in natural language.

The agent automatically routes each request into one of three execution modes:

### Research Mode

Tests historical market hypotheses using quantitative evidence.

Example:

```text
Do downward wind forecast revisions increase the probability that ID1 > ID3?
```

Research Mode is intended for hypothesis testing, exploratory analysis, robustness checks, and the development of market signals.

---

### Historical Replay Mode

Reconstructs what the system could have concluded at a past decision time using **only information that was available at that time**.

Example:

```text
What would the agent have concluded at 08:00 on 2025-06-12?
```

This mode is designed to prevent future-information leakage and support realistic historical evaluation.

---

### Live Forecast Mode

Uses the latest available market and forecast data to evaluate the current market state.

Example:

```text
What does the latest wind forecast revision imply for tonight's intraday market?
```

The current prototype produces research-oriented market signals rather than executing trades.

---

# Core idea

The system separates six different responsibilities:

```text
Natural-language request
        ↓
Request routing
        ↓
Structured research specification
        ↓
Validation + point-in-time controls
        ↓
Quantitative execution
        ↓
Evidence interpretation
        ↓
Research conclusion
```

The LLM is not responsible for calculating market statistics.

Quantitative calculations are performed by deterministic tools. The reasoning layer interprets the resulting evidence and explains the research conclusion.

This separation is intended to make the workflow more reproducible, auditable, and suitable for quantitative market research.

---

# Architecture

The system is organized around a reusable agent core and market-specific quantitative components.

```text
                         Natural Language
                                │
                                ▼
                         Request Router
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
        Research Mode     Historical Replay    Live Forecast
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                    Structured Research Spec
                                │
                                ▼
                    Compiler + Validation
                                │
                                ▼
                  Point-in-Time Data Controls
                                │
                                ▼
                      Quantitative Engine
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
             Generic Quant Tools      Market-Specific
                                      Components
                     │                     │
                     │                 Germany DE-LU
                     │                 ├── data
                     │                 ├── features
                     │                 ├── targets
                     │                 ├── products
                     │                 └── market rules
                     │
                     └──────────┬──────────┘
                                │
                                ▼
                             Evidence
                                │
                                ▼
                       LLM Interpretation
                                │
                                ▼
                      Research Conclusion
```

The current prototype is implemented for **Germany DE-LU**.

The longer-term architecture is intended to support additional market-specific modules, for example:

```text
Market Interface
├── Germany DE-LU
├── France
└── China
```

The reusable agent workflow should remain largely unchanged, while market-specific data sources, products, rules, features, and targets can be replaced or extended.

---

# Quantitative research layer

The next stage of development focuses on strengthening the quantitative layer behind the agent.

The intended research workflow is:

```text
Point-in-time market data
        ↓
Feature construction
        ↓
Target construction
        ↓
Statistical analysis
        ↓
Signal development
        ↓
Walk-forward evaluation
        ↓
Backtesting
        ↓
Trading-oriented research metrics
```

Initial Germany-focused research includes variables such as:

- wind forecast revisions
- solar forecast revisions
- load forecasts
- residual load
- day-ahead prices
- intraday prices
- generation
- forecast vintages

A key objective is to test whether information available at a given decision time contains predictive value for subsequent intraday price movements.

---

# Point-in-time correctness

Point-in-time correctness is a central design principle.

For every historical research or replay task, the system should distinguish between:

```text
delivery time
decision time
data publication time
forecast vintage
actual outcome
```

Historical execution must only use information that would genuinely have been available at the specified decision time.

For example:

```text
08:00 decision time
        ↓
latest forecast available before 08:00
        ↓
market information available before 08:00
        ↓
research / signal
        ↓
future market outcome
```

The future outcome may be used later for evaluation, but it must not leak into the research process itself.

This is particularly important for:

- forecast revision studies
- historical replay
- signal development
- machine-learning validation
- backtesting

---

# Current Germany DE-LU data work

The project is being developed alongside a separate point-in-time data pipeline for German power-market research.

Current work includes:

- ENTSO-E wind forecast vintages
- ENTSO-E solar forecast vintages
- day-ahead forecasts
- intraday/current forecasts
- historical actual generation
- timestamped data collection
- change detection
- 15-minute time series
- market-price data exploration and alignment

The goal is to preserve not only the final value of a forecast, but also **which version of the forecast was available at each point in time**.

This allows the research layer to reconstruct realistic historical information sets.

---

# Repository structure

Main components:

```text
routing/
    Request understanding and execution-mode selection

compiler/
    Research-spec compilation
    Validation
    Variable registry

execution/
    Research execution
    Historical replay
    Live forecast workflows

tools/
    Data access
    Quantitative analysis tools

reasoning/
    Interpretation of quantitative evidence

schemas/
    Structured research specifications

ui/
    Streamlit demo interface

tests/
    Automated validation and parsing tests
```

As the project develops, the quantitative layer is intended to become more explicitly separated into reusable quantitative tools and market-specific components.

Conceptually:

```text
quant/
    statistics
    hypothesis tests
    robustness
    walk-forward evaluation
    backtesting
    metrics

markets/
    de_lu/
        data
        features
        targets
        products
        market rules
```

---

# Run the demo

Requires **Python 3.10+**.

Install runtime dependencies:

```bash
python -m pip install -r requirements_runtime.txt
```

Launch the Streamlit interface:

```bash
streamlit run ui/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

The interface accepts natural-language questions and automatically routes each request to:

- Research
- Historical Replay
- Live Forecast

The three buttons shown in the interface are only demo shortcuts.

They prefill example questions; the router still determines the execution mode automatically.

---

# Example requests

## Research

```text
Do downward wind revisions increase the probability that ID1 > ID3?
```

## Historical Replay

```text
What would the agent have concluded at 08:00 on 2025-06-12?
```

## Live Forecast

```text
What does the latest wind forecast revision imply for tonight's intraday market?
```

---

# Market data

The current prototype is designed around German DE-LU power-market data.

Relevant variables include:

- day-ahead prices
- intraday prices
- wind forecasts
- solar forecasts
- generation
- load
- forecast revisions

Historical and live execution can use the configured market dataset through:

```text
MARKET_DATA_CSV
```

Private or licensed market datasets are not included in this repository.

---

# Environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Available configuration:

```text
OPENAI_API_KEY=
MARKET_DATA_CSV=
```

Do not commit `.env`.

The repository excludes `.env` through `.gitignore`.

---

# Run tests

Install development dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python -m pytest
```

The current test suite includes validation of delivery-window and decision-time parsing used by the structured research workflow.

---

# Design principles

## 1. Structured before generative

Natural-language requests are converted into structured research specifications before quantitative execution.

The LLM should not directly invent market variables, time boundaries, or numerical results.

---

## 2. Point-in-time correctness

Historical analysis must respect the information that was genuinely available at the stated decision time.

This reduces look-ahead bias and future-information leakage.

---

## 3. Deterministic quantitative tools

Statistical calculations and data transformations are performed by explicit quantitative tools.

The reasoning layer interprets their output rather than generating numerical evidence itself.

---

## 4. Separation of reusable logic and market knowledge

The agent workflow should not depend on one specific electricity market.

Market-specific elements such as:

- data sources
- product definitions
- trading windows
- market rules
- features
- targets

should be isolated from reusable research and quantitative logic.

Germany DE-LU is the first implementation.

---

## 5. Evidence-based conclusions

Research conclusions should be grounded in the quantitative evidence generated by the system.

The goal is not to produce a plausible narrative, but to determine what the available data actually supports.

---

# Current development direction

The hackathon prototype established the main agent architecture and end-to-end execution workflow.

Development is continuing beyond the hackathon.

The current priority is the **Germany-focused quantitative research layer**, including:

```text
forecast vintages
        ↓
feature engineering
        ↓
statistical relationships
        ↓
price / spread prediction
        ↓
signal development
        ↓
walk-forward validation
        ↓
backtesting
```

Future extensions may include:

- stronger intraday price modelling
- machine-learning forecasting
- trading-signal evaluation
- transaction-cost-aware backtesting
- BESS optimization
- additional European electricity markets
- market-specific modules for France and other regions

---

# Project scope

This repository originated from the **X-IA Hackathon — Rise of Agents X**.

The hackathon version demonstrated:

- natural-language request routing
- Research Mode
- Historical Replay Mode
- Live Forecast Mode
- structured research specifications
- decision-time constraints
- point-in-time execution logic
- automated validation tests
- Streamlit demo interface

The project is now being extended into a broader electricity-market quantitative research framework.

It remains a **research prototype** and is not currently a production trading or execution system.
