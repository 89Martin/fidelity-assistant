# Fidelity Wealth & Debt Assistant

A local, professional Streamlit app for a small-balance Fidelity investor
($250 start, $5–$25/week) running a dual-track plan: long-term retirement
growth **and** a 3–5 year debt-payoff lump sum.

## Features

- **Live prices** for Fidelity Zero-fee funds (FZROX, FNILX) via `yfinance`.
- **Deterministic Python math engine** — a strict weekly loop computes 5-year
  compound growth at a conservative 8% annual return. The AI never does math.
- **Debt payoff outlook** — months-to-zero and interest paid, interest-aware.
- **AI fiduciary strategist** — Claude (claude-3-5-sonnet) returns a
  *Core & Satellite* allocation (FZROX/FNILX for retirement, SPAXX for the debt
  lump sum), grounded in your profile and the live prices.
- Secure key handling via `python-dotenv` + `.env`.

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. add your Anthropic API key
copy .env.example .env          # Windows
# cp .env.example .env           # macOS/Linux
# then edit .env and paste your real key

# 4. run
streamlit run app.py
```

The app opens at http://localhost:8501.

## Notes

- FZROX/FNILX are mutual funds and price once per trading day; the app shows the
  most recent close.
- Educational tool only — not financial advice. Index funds carry market risk;
  SPAXX is not FDIC-insured.
```
