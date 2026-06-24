"""
Fidelity Wealth & Debt Assistant — Dark Dashboard Edition
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

try:
    from anthropic import Anthropic, APIError
except ImportError:
    Anthropic = None
    APIError = Exception

# ── Config ──────────────────────────────────────────────────────────────────

load_dotenv()

APP_TITLE = "Wealth & Debt Assistant"
CORE_TICKERS: Dict[str, str] = {
    "FZROX": "Total Market",
    "FNILX": "Large Cap",
}
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
ANNUAL_RETURN    = 0.08
PROJECTION_YEARS = 5
WEEKS_PER_YEAR   = 52

# ── Theme colors (mirrors the screenshot palette) ────────────────────────────

DARK_BG      = "#0d0f1c"
CARD_BG      = "#13152a"
CARD_BORDER  = "#1e2140"
ACCENT       = "#7b5cf0"
ACCENT2      = "#a78bfa"
GREEN        = "#22c55e"
RED          = "#ef4444"
TEXT_PRIMARY = "#f0f2ff"
TEXT_MUTED   = "#8b8fb5"

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class InvestorProfile:
    starting_balance: float
    weekly_deposit: float
    debt_balance: float
    debt_apr: float
    timeline_years: int
    annual_return: float = ANNUAL_RETURN


@dataclass
class FundQuote:
    ticker: str
    name: str
    price: Optional[float]
    as_of: Optional[str]
    error: Optional[str] = None


# ── Math engine ──────────────────────────────────────────────────────────────

def project_growth(profile: InvestorProfile) -> pd.DataFrame:
    weekly_rate = (1.0 + profile.annual_return) ** (1.0 / WEEKS_PER_YEAR) - 1.0
    total_weeks = profile.timeline_years * WEEKS_PER_YEAR
    balance = float(profile.starting_balance)
    contributed = float(profile.starting_balance)
    start = dt.date.today()

    rows = []
    for week in range(1, total_weeks + 1):
        balance += profile.weekly_deposit
        contributed += profile.weekly_deposit
        balance *= 1.0 + weekly_rate
        rows.append({
            "Week": week,
            "Date": start + dt.timedelta(weeks=week),
            "Balance": round(balance, 2),
            "Contributed": round(contributed, 2),
            "Growth": round(balance - contributed, 2),
        })
    return pd.DataFrame(rows).set_index("Date")


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in range(1, PROJECTION_YEARS + 1):
        sl = df[df["Week"] <= yr * WEEKS_PER_YEAR]
        if sl.empty:
            continue
        r = sl.iloc[-1]
        rows.append({"Year": yr, "Balance": float(r["Balance"]),
                     "Contributed": float(r["Contributed"]), "Growth": float(r["Growth"])})
    return pd.DataFrame(rows)


def debt_payoff(debt: float, apr: float, weekly_pmt: float) -> Tuple[Optional[int], float]:
    if debt <= 0:
        return 0, 0.0
    monthly_rate = apr / 12.0
    monthly_pmt  = weekly_pmt * (WEEKS_PER_YEAR / 12.0)
    bal   = float(debt)
    interest_paid = 0.0
    months = 0
    while bal > 0 and months < 600:
        interest = bal * monthly_rate
        if monthly_pmt <= interest and monthly_rate > 0:
            return None, interest_paid
        interest_paid += interest
        bal = bal + interest - monthly_pmt
        months += 1
    return months, round(interest_paid, 2)


# ── Market data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_quote(ticker: str, name: str) -> FundQuote:
    try:
        tkr  = yf.Ticker(ticker)
        price: Optional[float] = None
        as_of: Optional[str]   = None
        fast = getattr(tkr, "fast_info", None)
        if fast is not None:
            last = fast.get("last_price") if hasattr(fast, "get") else None
            if last:
                price = float(last)
        if price is None:
            hist = tkr.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1])
                as_of  = str(hist.index[-1].date())
        as_of = as_of or str(dt.date.today())
        if price is None:
            return FundQuote(ticker, name, None, as_of, "No price data.")
        return FundQuote(ticker, name, round(price, 2), as_of)
    except Exception as exc:
        return FundQuote(ticker, name, None, str(dt.date.today()), str(exc))


def fetch_quotes() -> List[FundQuote]:
    return [fetch_quote(t, n) for t, n in CORE_TICKERS.items()]


# ── AI ───────────────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        pass
    return os.getenv("ANTHROPIC_API_KEY")


def build_prompt(profile: InvestorProfile, quotes: List[FundQuote]) -> str:
    quote_lines = [
        f"- {q.ticker} ({q.name}): ${q.price:,.2f} as of {q.as_of}"
        if q.price else f"- {q.ticker}: unavailable"
        for q in quotes
    ]
    return f"""You are a strict, conservative fiduciary financial strategist.

CLIENT PROFILE
- Starting balance: ${profile.starting_balance:,.2f}
- Weekly deposit: ${profile.weekly_deposit:,.2f}
- Outstanding debt: ${profile.debt_balance:,.2f} at {profile.debt_apr*100:.1f}% APR
- Primary goal: pay off debt within {profile.timeline_years} years
- Secondary goal: long-term retirement growth

LIVE FUND PRICES:
{chr(10).join(quote_lines)}

INSTRUMENTS: FZROX (Total Market, retirement), FNILX (Large Cap, retirement), SPAXX (money market, debt lump sum)

Recommend a weekly split of ${profile.weekly_deposit:,.2f} across FZROX, FNILX, SPAXX.
Percentages must sum to 100%. Weight toward debt elimination given the {profile.debt_apr*100:.1f}% APR.

OUTPUT (use this exact structure):
1. **Allocation Table** — % and $ for each fund
2. **Rationale** — 3-4 sentences
3. **Milestones** — what to adjust once debt is paid
4. **Disclaimer** — one plain-language line

Keep it under 350 words. Do NOT calculate compound projections."""


def generate_plan(profile: InvestorProfile, quotes: List[FundQuote]) -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found. Add it in Streamlit → Settings → Secrets.")
    if Anthropic is None:
        raise RuntimeError("Run: pip install anthropic")
    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        temperature=0.2,
        system="You are a careful fiduciary. Be precise, conservative, and concise.",
        messages=[{"role": "user", "content": build_prompt(profile, quotes)}],
    )
    return "\n".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


# ── CSS ──────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── Page background ── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] > div:first-child {{
        background: {CARD_BG};
        border-right: 1px solid {CARD_BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}
    [data-testid="stSidebar"] label {{ color: {TEXT_MUTED} !important; font-size: 0.78rem !important; }}

    /* ── All inputs / sliders ── */
    .stSlider > div[data-baseweb] > div {{ background: {CARD_BORDER}; }}
    .stSlider [data-testid="stThumbValue"] {{ color: {ACCENT2} !important; }}
    input[type="number"] {{
        background: {CARD_BG} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] > div {{
        background: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
    [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY} !important; font-size: 1.7rem !important; font-weight: 700 !important; }}
    [data-testid="stMetricDelta"] {{ font-size: 0.82rem !important; }}

    /* ── Section headings ── */
    h2 {{ color: {TEXT_PRIMARY} !important; font-size: 1.05rem !important; font-weight: 600 !important;
          text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem !important; }}

    /* ── Dividers ── */
    hr {{ border-color: {CARD_BORDER} !important; }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{ opacity: 0.88 !important; }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
    }}
    [data-testid="stExpander"] summary {{ color: {TEXT_MUTED} !important; font-size: 0.85rem !important; }}

    /* ── Tables ── */
    table {{ background: {CARD_BG} !important; color: {TEXT_PRIMARY} !important; border-radius: 10px; overflow: hidden; }}
    th {{ background: {CARD_BORDER} !important; color: {TEXT_MUTED} !important; font-size: 0.75rem !important; text-transform: uppercase; }}
    td {{ border-color: {CARD_BORDER} !important; }}

    /* ── Alert/info boxes ── */
    .stAlert {{ background: {CARD_BG} !important; border-left: 3px solid {ACCENT} !important; border-radius: 10px; color: {TEXT_PRIMARY} !important; }}

    /* ── AI response card ── */
    .ai-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        color: {TEXT_PRIMARY};
        line-height: 1.7;
    }}
    .ai-card h1, .ai-card h2, .ai-card h3,
    .ai-card h4 {{ color: {ACCENT2} !important; }}
    .ai-card strong {{ color: {TEXT_PRIMARY}; }}

    /* ── Fund badge ── */
    .fund-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }}
    .fund-ticker {{ font-size: 1.05rem; font-weight: 700; color: {ACCENT2}; }}
    .fund-name   {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .fund-price  {{ font-size: 1.9rem; font-weight: 800; color: {TEXT_PRIMARY}; }}
    .fund-date   {{ font-size: 0.7rem; color: {TEXT_MUTED}; margin-top: 0.2rem; }}

    /* ── Page header ── */
    .page-header {{
        display: flex; align-items: center; gap: 0.75rem;
        margin-bottom: 1.5rem; padding-bottom: 1rem;
        border-bottom: 1px solid {CARD_BORDER};
    }}
    .page-title {{ font-size: 1.6rem; font-weight: 800; color: {TEXT_PRIMARY}; margin: 0; }}
    .page-sub   {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin: 0; }}
    .dot        {{ width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; display:inline-block; }}

    /* ── Caption / footnote ── */
    [data-testid="stCaptionContainer"] p {{ color: {TEXT_MUTED} !important; font-size: 0.72rem !important; }}

    /* ── Hide Streamlit default chrome ── */
    #MainMenu, footer, [data-testid="stToolbar"] {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_MUTED, family="Inter, sans-serif", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=CARD_BORDER,
        borderwidth=1,
        font=dict(color=TEXT_MUTED, size=10),
    ),
    xaxis=dict(gridcolor=CARD_BORDER, linecolor=CARD_BORDER, zeroline=False),
    yaxis=dict(gridcolor=CARD_BORDER, linecolor=CARD_BORDER, zeroline=False,
               tickprefix="$", tickformat=",.0f"),
)


def growth_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Balance"],
        name="Portfolio Value",
        line=dict(color=ACCENT, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(123,92,240,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Balance: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Contributed"],
        name="Amount Invested",
        line=dict(color=ACCENT2, width=1.8, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Contributed: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    return fig


def allocation_donut(labels: List[str], values: List[float]) -> go.Figure:
    colors = [ACCENT, ACCENT2, "#06b6d4"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="Inter, sans-serif"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> InvestorProfile:
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1rem 0 1.5rem 0; text-align:center;'>
            <span style='font-size:1.8rem;'>💵</span>
            <p style='margin:0.3rem 0 0 0; font-size:0.95rem; font-weight:700; color:{TEXT_PRIMARY};'>Fidelity AI Assistant</p>
            <p style='margin:0; font-size:0.7rem; color:{TEXT_MUTED};'>Wealth & Debt Planner</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;'>Portfolio</p>", unsafe_allow_html=True)
        starting_balance = st.number_input("Starting balance ($)", min_value=0.0, max_value=1_000_000.0, value=250.0, step=25.0)
        weekly_deposit   = st.slider("Weekly deposit ($)", min_value=5, max_value=25, value=15, step=1)

        st.divider()
        st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;'>Debt</p>", unsafe_allow_html=True)
        debt_balance  = st.number_input("Outstanding debt ($)", min_value=0.0, max_value=1_000_000.0, value=2_000.0, step=100.0)
        debt_apr_pct  = st.slider("APR (%)", min_value=0.0, max_value=35.0, value=22.0, step=0.5)
        timeline_years = st.select_slider("Payoff timeline (years)", options=[3, 4, 5], value=4)

        st.divider()
        st.caption("Projections use 8% annual return, compounded weekly. Math is computed in Python — not by AI.")

    return InvestorProfile(
        starting_balance=float(starting_balance),
        weekly_deposit=float(weekly_deposit),
        debt_balance=float(debt_balance),
        debt_apr=float(debt_apr_pct) / 100.0,
        timeline_years=int(timeline_years),
    )


# ── Main layout ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Fidelity Wealth & Debt Assistant",
        page_icon="💵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    profile = render_sidebar()

    # ── Page header
    st.markdown(f"""
    <div class="page-header">
        <span class="dot"></span>
        <div>
            <p class="page-title">{APP_TITLE}</p>
            <p class="page-sub">Dual-track plan · Retirement growth + 3–5 year debt payoff</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live prices row
    st.markdown("## Live Fund Prices")
    quotes = fetch_quotes()
    price_cols = st.columns(len(quotes) + 1)
    for col, q in zip(price_cols, quotes):
        with col:
            price_str = f"${q.price:,.2f}" if q.price else "N/A"
            error_str = f"<p style='color:{RED}; font-size:0.7rem;'>{q.error}</p>" if q.error else f"<p class='fund-date'>Last close · {q.as_of}</p>"
            st.markdown(f"""
            <div class="fund-card">
                <p class="fund-name">{q.name}</p>
                <p class="fund-ticker">{q.ticker}</p>
                <p class="fund-price">{price_str}</p>
                {error_str}
            </div>
            """, unsafe_allow_html=True)

    # SPAXX static badge
    with price_cols[-1]:
        st.markdown(f"""
        <div class="fund-card">
            <p class="fund-name">Money Market</p>
            <p class="fund-ticker">SPAXX</p>
            <p class="fund-price" style="font-size:1.2rem; padding-top:0.55rem;">Debt lump sum vehicle</p>
            <p class="fund-date">Safe · ~4.5-5% yield</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Top KPI metrics
    growth_df = project_growth(
        InvestorProfile(
            starting_balance=profile.starting_balance,
            weekly_deposit=profile.weekly_deposit,
            debt_balance=profile.debt_balance,
            debt_apr=profile.debt_apr,
            timeline_years=PROJECTION_YEARS,
        )
    )
    final = growth_df.iloc[-1]
    total_weeks = PROJECTION_YEARS * WEEKS_PER_YEAR
    total_invested = profile.starting_balance + profile.weekly_deposit * total_weeks

    weekly_to_debt = profile.weekly_deposit * 0.60
    months, interest_paid = debt_payoff(profile.debt_balance, profile.debt_apr, weekly_to_debt)
    debt_free_str = f"{months} mo ({months/12:.1f} yr)" if months is not None else "Never"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("5-Year Portfolio Value",  f"${float(final['Balance']):,.0f}")
    k2.metric("Total Market Gains",      f"${float(final['Growth']):,.0f}")
    k3.metric("Total Invested",          f"${total_invested:,.0f}")
    k4.metric("Est. Months to Debt-Free", debt_free_str)

    st.divider()

    # ── Chart + donut side by side
    st.markdown("## 5-Year Compound Growth")
    chart_col, donut_col = st.columns([3, 1])

    with chart_col:
        st.plotly_chart(growth_chart(growth_df), use_container_width=True, config={"displayModeBar": False})

    with donut_col:
        st.markdown("#### Illustrative Allocation")
        # Default split for the donut (will be replaced by AI recommendation)
        st.plotly_chart(
            allocation_donut(["FZROX", "FNILX", "SPAXX"], [30.0, 10.0, 60.0]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("60% → debt payoff (SPAXX)\n30% → Total Market (FZROX)\n10% → Large Cap (FNILX)")

    # Year-by-year summary
    with st.expander("Year-by-year breakdown"):
        summary = annual_summary(growth_df)
        display = summary.copy()
        for c in ("Balance", "Contributed", "Growth"):
            display[c] = display[c].map(lambda v: f"${v:,.2f}")
        st.table(display.set_index("Year"))

    st.divider()

    # ── Debt panel
    st.markdown("## Debt Payoff Outlook")
    if profile.debt_balance <= 0:
        st.success("No debt — 100% of deposits can go toward retirement. 🎉")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Debt Balance",           f"${profile.debt_balance:,.2f}")
        d2.metric("APR",                    f"{profile.debt_apr*100:.1f}%")
        d3.metric("Interest Cost (est.)",   f"${interest_paid:,.2f}")

        if months is None:
            st.warning("Current payment doesn't outpace interest. Increase weekly deposit or debt allocation.")
        elif months / 12.0 > profile.timeline_years:
            st.info(f"At 60% toward debt you'd pay off in {months/12:.1f} years — above your {profile.timeline_years}-year target. The AI plan below will weight more toward SPAXX.")
        else:
            st.success(f"On track to be debt-free in {months} months ({months/12:.1f} years) within your {profile.timeline_years}-year goal. ✅")

    st.divider()

    # ── AI panel
    st.markdown("## AI Allocation Strategy")
    st.caption("Claude acts as a strict fiduciary — it sees your profile and the live prices, then recommends a Core & Satellite split.")

    if st.button("✦  Generate AI Allocation Plan", use_container_width=True):
        with st.spinner("Consulting the AI fiduciary..."):
            try:
                plan = generate_plan(profile, quotes)
                st.session_state["ai_plan"] = plan
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Error: {exc}")

    if st.session_state.get("ai_plan"):
        st.markdown(f'<div class="ai-card">{st.session_state["ai_plan"]}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("Educational tool only — not financial advice. Index funds carry market risk. SPAXX is not FDIC-insured. Verify all figures with Fidelity.")


if __name__ == "__main__":
    main()
