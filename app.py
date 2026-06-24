"""
Fidelity Wealth & Debt Assistant
================================

A local Streamlit application that helps a small-balance investor at Fidelity
plan a dual-track strategy:

    1. Long-term retirement growth via Fidelity Zero-fee index funds
       (FZROX - Total Market, FNILX - Large Cap).
    2. A 3-5 year debt-payoff lump sum built in a safe cash vehicle
       (SPAXX money market).

Design principles
-----------------
* All financial math is computed natively in Python (a strict, deterministic
  loop). The AI is never asked to "do math".
* Real-time prices are pulled from yfinance.
* Claude (claude-3-5-sonnet) acts only as a fiduciary strategist, given the
  user's profile and live prices, and returns a Core & Satellite allocation.
* The ANTHROPIC_API_KEY is loaded securely from a local .env file.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# The Anthropic SDK is imported lazily-friendly here; we still import at top so
# any environment issues surface immediately.
try:
    from anthropic import Anthropic, APIError
except ImportError:  # pragma: no cover - guidance for first-time setup
    Anthropic = None  # type: ignore[assignment]
    APIError = Exception  # type: ignore[assignment,misc]


# --------------------------------------------------------------------------- #
# Configuration & constants
# --------------------------------------------------------------------------- #

load_dotenv()  # Loads ANTHROPIC_API_KEY from a local .env file.

APP_TITLE: str = "Fidelity Wealth & Debt Assistant"
APP_ICON: str = "💵"

# Fidelity Zero-fee funds (mutual funds priced once per day by yfinance).
CORE_TICKERS: Dict[str, str] = {
    "FZROX": "Fidelity ZERO Total Market Index",
    "FNILX": "Fidelity ZERO Large Cap Index",
}

ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

CONSERVATIVE_ANNUAL_RETURN: float = 0.08  # 8% conservative market assumption.
PROJECTION_YEARS: int = 5
WEEKS_PER_YEAR: int = 52


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class InvestorProfile:
    """Typed snapshot of everything the AI and math engine need."""

    starting_balance: float
    weekly_deposit: float
    debt_balance: float
    debt_apr: float
    timeline_years: int
    annual_return: float = CONSERVATIVE_ANNUAL_RETURN


@dataclass
class FundQuote:
    """A single fund's live (or last-close) price information."""

    ticker: str
    name: str
    price: Optional[float]
    currency: str
    as_of: Optional[str]
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Python math engine (deterministic — never delegated to the AI)
# --------------------------------------------------------------------------- #


def project_compound_growth(profile: InvestorProfile) -> pd.DataFrame:
    """
    Compute week-by-week compound growth using a strict Python loop.

    Interest is compounded weekly using the equivalent weekly rate derived from
    the annual return. The weekly deposit is added at the start of each week,
    then growth is applied. Returns a DataFrame indexed by week with running
    balance, total contributed, and cumulative growth.
    """
    weekly_rate: float = (1.0 + profile.annual_return) ** (1.0 / WEEKS_PER_YEAR) - 1.0
    total_weeks: int = profile.timeline_years * WEEKS_PER_YEAR

    balance: float = float(profile.starting_balance)
    contributed: float = float(profile.starting_balance)

    weeks: List[int] = []
    dates: List[dt.date] = []
    balances: List[float] = []
    contributions: List[float] = []
    growths: List[float] = []

    start_date: dt.date = dt.date.today()

    for week in range(1, total_weeks + 1):
        # Deposit first, then apply that week's compounded growth.
        balance += profile.weekly_deposit
        contributed += profile.weekly_deposit
        balance *= 1.0 + weekly_rate

        weeks.append(week)
        dates.append(start_date + dt.timedelta(weeks=week))
        balances.append(round(balance, 2))
        contributions.append(round(contributed, 2))
        growths.append(round(balance - contributed, 2))

    return pd.DataFrame(
        {
            "Week": weeks,
            "Date": dates,
            "Balance": balances,
            "Contributed": contributions,
            "Growth": growths,
        }
    ).set_index("Date")


def annual_summary(growth_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the weekly projection to a clean year-end summary table."""
    rows: List[Dict[str, object]] = []
    for year in range(1, PROJECTION_YEARS + 1):
        week_marker: int = year * WEEKS_PER_YEAR
        year_slice = growth_df[growth_df["Week"] <= week_marker]
        if year_slice.empty:
            continue
        end_row = year_slice.iloc[-1]
        rows.append(
            {
                "Year": year,
                "Balance": float(end_row["Balance"]),
                "Contributed": float(end_row["Contributed"]),
                "Growth": float(end_row["Growth"]),
            }
        )
    return pd.DataFrame(rows)


def debt_payoff_estimate(
    debt_balance: float, debt_apr: float, weekly_toward_debt: float
) -> Tuple[Optional[int], float]:
    """
    Estimate months to wipe out debt given a fixed weekly payment, accounting
    for interest accrual. Returns (months_to_payoff, total_interest_paid).

    months_to_payoff is None if the payment never covers the accruing interest.
    """
    if debt_balance <= 0:
        return 0, 0.0

    monthly_rate: float = debt_apr / 12.0
    monthly_payment: float = weekly_toward_debt * (WEEKS_PER_YEAR / 12.0)

    balance: float = float(debt_balance)
    total_interest: float = 0.0
    months: int = 0
    max_months: int = 600  # 50-year safety cap to avoid infinite loops.

    while balance > 0 and months < max_months:
        interest: float = balance * monthly_rate
        # If the payment can't outpace interest, payoff is impossible.
        if monthly_payment <= interest and monthly_rate > 0:
            return None, total_interest
        total_interest += interest
        balance = balance + interest - monthly_payment
        months += 1

    return months, round(total_interest, 2)


# --------------------------------------------------------------------------- #
# Market data (yfinance)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fund_quote(ticker: str, name: str) -> FundQuote:
    """
    Fetch the most recent price for a fund via yfinance. Mutual funds update
    once per trading day, so we fall back to the last close. Cached for 15 min.
    """
    try:
        tkr = yf.Ticker(ticker)
        price: Optional[float] = None
        as_of: Optional[str] = None
        currency: str = "USD"

        # Prefer fast_info (lightweight), fall back to recent history.
        fast = getattr(tkr, "fast_info", None)
        if fast is not None:
            last = fast.get("last_price") if hasattr(fast, "get") else None
            if last:
                price = float(last)
                currency = (fast.get("currency") if hasattr(fast, "get") else None) or "USD"

        if price is None:
            hist = tkr.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1])
                as_of = str(hist.index[-1].date())

        if as_of is None:
            as_of = str(dt.date.today())

        if price is None:
            return FundQuote(ticker, name, None, currency, as_of, "No price data returned.")

        return FundQuote(ticker, name, round(price, 2), currency, as_of, None)

    except Exception as exc:  # noqa: BLE001 - surface any data error to the UI.
        return FundQuote(ticker, name, None, "USD", str(dt.date.today()), str(exc))


def fetch_all_quotes() -> List[FundQuote]:
    """Fetch quotes for every core ticker."""
    return [fetch_fund_quote(t, n) for t, n in CORE_TICKERS.items()]


# --------------------------------------------------------------------------- #
# AI strategy module (Claude as fiduciary)
# --------------------------------------------------------------------------- #


def build_strategy_prompt(profile: InvestorProfile, quotes: List[FundQuote]) -> str:
    """Construct a structured, fact-grounded prompt for the AI strategist."""
    quote_lines: List[str] = []
    for q in quotes:
        if q.price is not None:
            quote_lines.append(f"- {q.ticker} ({q.name}): ${q.price:,.2f} as of {q.as_of}")
        else:
            quote_lines.append(f"- {q.ticker} ({q.name}): price unavailable ({q.error})")
    quotes_block: str = "\n".join(quote_lines)

    return f"""You are a strict, conservative fiduciary financial strategist. You are
NOT a licensed advisor and must include a brief disclaimer, but you must act in
the client's best interest, prioritizing capital preservation and debt freedom.

CLIENT PROFILE
- Brokerage: Fidelity
- Current invested/cash balance: ${profile.starting_balance:,.2f}
- Weekly deposit: ${profile.weekly_deposit:,.2f}
- Outstanding debt: ${profile.debt_balance:,.2f} at {profile.debt_apr * 100:.1f}% APR
- Primary timeline: {profile.timeline_years} years (debt payoff is the priority)
- Secondary goal: long-term retirement growth

LIVE FUND PRICES (already fetched — do not estimate prices):
{quotes_block}

AVAILABLE INSTRUMENTS
- FZROX: Fidelity ZERO Total Market Index (broad market, retirement core)
- FNILX: Fidelity ZERO Large Cap Index (large-cap tilt, retirement core)
- SPAXX: Fidelity Government Money Market (safe cash vehicle for the debt lump sum)

YOUR TASK
Recommend a "Core & Satellite" weekly allocation of the ${profile.weekly_deposit:,.2f}
deposit that:
  1. CORE (retirement): a percentage into broad-market index funds (FZROX/FNILX).
  2. SATELLITE / SAFETY (debt payoff): a percentage into SPAXX to build a lump
     sum that can wipe out the debt within the {profile.timeline_years}-year window.

Because the debt carries a {profile.debt_apr * 100:.1f}% guaranteed cost, weight the
allocation toward debt elimination first, then scale retirement contributions
up as the debt shrinks.

OUTPUT FORMAT (use this exact structure):
1. **Allocation Table** — percentage and weekly dollar amount for each of:
   FZROX, FNILX, SPAXX. Percentages must sum to 100%.
2. **Rationale** — 3-4 sentences explaining the split given the debt APR.
3. **Milestones** — what to change once the debt is paid off.
4. **Disclaimer** — one line, plain language.

Do not perform long-horizon compound projections (the app already computes those
in Python). Keep the response under 350 words."""


def get_api_key() -> Optional[str]:
    """
    Resolve the Anthropic API key from either source, in order:
      1. Streamlit Cloud secrets (st.secrets) — used when deployed.
      2. A local .env / environment variable — used when running locally.
    """
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:  # noqa: BLE001 - st.secrets raises if no secrets file exists.
        pass
    return os.getenv("ANTHROPIC_API_KEY")


def generate_ai_allocation(profile: InvestorProfile, quotes: List[FundQuote]) -> str:
    """Call Claude to produce a Core & Satellite allocation plan."""
    api_key: Optional[str] = get_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Locally: create a .env file with "
            "ANTHROPIC_API_KEY=sk-ant-... — On Streamlit Cloud: add it under "
            "App settings → Secrets."
        )
    if Anthropic is None:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        )

    client = Anthropic(api_key=api_key)
    prompt: str = build_strategy_prompt(profile, quotes)

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        temperature=0.2,
        system="You are a careful fiduciary. Be precise, conservative, and concise.",
        messages=[{"role": "user", "content": prompt}],
    )

    # Concatenate any text blocks in the response.
    parts: List[str] = [block.text for block in message.content if getattr(block, "type", "") == "text"]
    return "\n".join(parts).strip()


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #


def money(value: float) -> str:
    """Format a float as USD currency."""
    return f"${value:,.2f}"


def render_header() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0 1rem 0;">
            <h1 style="margin-bottom: 0;">{APP_ICON} {APP_TITLE}</h1>
            <p style="color: #4b7a3f; font-size: 1.05rem; margin-top: 0.25rem;">
                A dual-track plan: build retirement wealth while engineering a
                3-5 year debt-payoff lump sum.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> InvestorProfile:
    """Render input controls and return a typed InvestorProfile."""
    st.sidebar.header("⚙️ Your Profile")

    starting_balance = st.sidebar.number_input(
        "Starting balance ($)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=250.0,
        step=25.0,
        help="Your current Fidelity balance. Defaults to $250.",
    )

    weekly_deposit = st.sidebar.slider(
        "Weekly deposit ($)",
        min_value=5,
        max_value=25,
        value=15,
        step=1,
        help="How much you add every week ($5-$25).",
    )

    st.sidebar.divider()
    st.sidebar.subheader("💳 Debt")

    debt_balance = st.sidebar.number_input(
        "Outstanding debt ($)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=2_000.0,
        step=100.0,
    )

    debt_apr_pct = st.sidebar.slider(
        "Debt APR (%)",
        min_value=0.0,
        max_value=35.0,
        value=22.0,
        step=0.5,
        help="The annual interest rate on your debt.",
    )

    timeline_years = st.sidebar.select_slider(
        "Debt-payoff timeline (years)",
        options=[3, 4, 5],
        value=4,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Projections assume a conservative **8%** annual market return, "
        "compounded weekly. Math is computed in Python, not by the AI."
    )

    return InvestorProfile(
        starting_balance=float(starting_balance),
        weekly_deposit=float(weekly_deposit),
        debt_balance=float(debt_balance),
        debt_apr=float(debt_apr_pct) / 100.0,
        timeline_years=int(timeline_years),
    )


def render_market_panel(quotes: List[FundQuote]) -> None:
    st.subheader("📈 Live Fidelity Zero-Fee Fund Prices")
    cols = st.columns(len(quotes))
    for col, q in zip(cols, quotes):
        with col:
            if q.price is not None:
                col.metric(label=f"{q.ticker} — {q.name}", value=money(q.price))
                col.caption(f"As of {q.as_of}")
            else:
                col.metric(label=f"{q.ticker} — {q.name}", value="N/A")
                col.caption(f"⚠️ {q.error}")


def render_projection_panel(profile: InvestorProfile) -> pd.DataFrame:
    st.subheader("🧮 5-Year Compound Growth Projection")

    growth_df = project_compound_growth(
        InvestorProfile(
            starting_balance=profile.starting_balance,
            weekly_deposit=profile.weekly_deposit,
            debt_balance=profile.debt_balance,
            debt_apr=profile.debt_apr,
            timeline_years=PROJECTION_YEARS,  # Chart always shows 5 years.
        )
    )

    final_row = growth_df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Projected balance (5 yr)", money(float(final_row["Balance"])))
    c2.metric("Total contributed", money(float(final_row["Contributed"])))
    c3.metric("Market growth", money(float(final_row["Growth"])))

    st.line_chart(growth_df[["Balance", "Contributed"]], height=320)

    with st.expander("Year-by-year summary"):
        summary = annual_summary(growth_df)
        summary_display = summary.copy()
        for c in ("Balance", "Contributed", "Growth"):
            summary_display[c] = summary_display[c].map(money)
        st.table(summary_display.set_index("Year"))

    return growth_df


def render_debt_panel(profile: InvestorProfile) -> None:
    st.subheader("💳 Debt Payoff Outlook")
    if profile.debt_balance <= 0:
        st.success("No debt entered — you're free to focus fully on retirement. 🎉")
        return

    # Illustrative: assume 60% of the weekly deposit is routed to debt (SPAXX).
    weekly_to_debt = profile.weekly_deposit * 0.60
    months, total_interest = debt_payoff_estimate(
        profile.debt_balance, profile.debt_apr, weekly_to_debt
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Weekly toward debt (illustrative 60%)", money(weekly_to_debt))
    if months is None:
        c2.metric("Months to payoff", "Never")
        st.warning(
            "At this payment level the interest outpaces your payments. "
            "Increase the weekly deposit or the share routed to debt."
        )
    else:
        years = months / 12.0
        c2.metric("Est. months to payoff", f"{months} ({years:.1f} yr)")
        c3.metric("Est. interest paid", money(total_interest))
        if years > profile.timeline_years:
            st.info(
                f"This pace ({years:.1f} yr) exceeds your {profile.timeline_years}-year "
                "target. The AI plan below will weight more toward debt."
            )


def render_ai_panel(profile: InvestorProfile, quotes: List[FundQuote]) -> None:
    st.subheader("🤖 AI Allocation Strategy")
    st.caption(
        "Claude acts as a strict fiduciary and proposes a Core & Satellite split "
        "using your profile and the live prices above."
    )

    if st.button("Generate AI Allocation Plan", type="primary", use_container_width=True):
        with st.spinner("Consulting the AI fiduciary..."):
            try:
                plan = generate_ai_allocation(profile, quotes)
                st.session_state["ai_plan"] = plan
            except RuntimeError as exc:
                st.error(str(exc))
            except APIError as exc:  # type: ignore[misc]
                st.error(f"Anthropic API error: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")

    if st.session_state.get("ai_plan"):
        st.markdown(st.session_state["ai_plan"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    render_header()
    profile = render_sidebar()

    quotes = fetch_all_quotes()
    render_market_panel(quotes)
    st.divider()

    render_projection_panel(profile)
    st.divider()

    render_debt_panel(profile)
    st.divider()

    render_ai_panel(profile, quotes)

    st.divider()
    st.caption(
        "Educational tool only — not financial advice. Index funds carry market "
        "risk; SPAXX is not FDIC-insured. Verify all figures with Fidelity."
    )


if __name__ == "__main__":
    main()
