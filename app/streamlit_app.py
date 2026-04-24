# app/streamlit_app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import ASSETS, ALL_ASSETS, INITIAL_CAPITAL, RISK_PROFILES
from src.data_loader import get_prices, get_returns, summary_stats
from src.indicators import all_indicators, support_resistance
from src.backtester import run_backtest, walk_forward_backtest
from src.optimization.portfolio_optimizer import (
    equal_weight, inverse_vol, max_sharpe,
    min_volatility, hrp, optimize_for_profile,
)
from src.risk_manager import (
    portfolio_var_cvar, run_stress_tests, risk_budget_allocation,
)
from src.report import compare_strategies

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoTrade — Optimal Portfolio Allocation",
    page_icon="📈",
    layout="wide",
)
st.title("AlgoTrade — Optimal Portfolio Allocation")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuração")

    selected = st.multiselect(
        "Ativos",
        options=ALL_ASSETS,
        default=["SPY", "QQQ", "GLD", "TLT", "BTC-USD"],
    )
    col1, col2 = st.columns(2)
    start_date = col1.text_input("De", "2019-01-01")
    end_date   = col2.text_input("Até", "2024-12-31")

    strategy = st.selectbox(
        "Estratégia de optimização",
        ["Max Sharpe", "Min Volatility", "HRP",
         "Inverse Vol", "Equal Weight"],
    )
    freq = st.selectbox(
        "Rebalanceamento",
        ["monthly", "quarterly", "weekly"],
    )
    capital = st.number_input(
        "Capital inicial (EUR)", value=INITIAL_CAPITAL, step=1000
    )

    st.divider()
    st.subheader("Split por perfil de risco")
    low_pct = st.slider("Low (%)",    0, 100, 40, step=5)
    med_pct = st.slider("Medium (%)", 0, 100 - low_pct, 40, step=5)
    high_pct = 100 - low_pct - med_pct
    st.info(f"High: {high_pct}%")

    run_btn = st.button("Correr análise", type="primary",
                        use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_bt, tab_ta, tab_risk, tab_alloc, tab_wf, tab_explore = st.tabs([
    "Backtest", "Análise técnica", "Risk", "Alocação",
    "Walk-forward", "Explorar dados",
])

if not run_btn:
    for t in [tab_bt, tab_ta, tab_risk, tab_alloc, tab_wf, tab_explore]:
        with t:
            st.info("Configura os parâmetros na barra lateral e clica Correr análise.")
    st.stop()

if len(selected) < 3:
    st.error("Seleciona pelo menos 3 ativos.")
    st.stop()

# ── Carregar dados ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    prices  = get_prices(list(tickers), start=start, end=end)
    returns = get_returns(prices)
    return prices, returns

with st.spinner("A carregar dados..."):
    prices, returns = load_data(tuple(selected), start_date, end_date)

STRATEGY_MAP = {
    "Max Sharpe":     lambda p: max_sharpe(p.pct_change().dropna(), max_weight=0.40),
    "Min Volatility": lambda p: min_volatility(p.pct_change().dropna(), max_weight=0.40),
    "HRP":            lambda p: hrp(p.pct_change().dropna()),
    "Inverse Vol":    lambda p: inverse_vol(p.pct_change().dropna()),
    "Equal Weight":   lambda p: equal_weight(list(p.columns)),
}
weight_fn = STRATEGY_MAP[strategy]

with st.spinner("A correr backtest..."):
    main_res = run_backtest(
        prices, weight_fn=weight_fn,
        initial_capital=capital, rebalance_freq=freq,
    )
    prof_res = {
        p: run_backtest(
            prices,
            weight_fn=lambda px, pr=p: optimize_for_profile(px, pr),
            initial_capital=capital, rebalance_freq=freq,
        )
        for p in ["low", "medium", "high"]
    }

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Backtest
# ════════════════════════════════════════════════════════════════════════════
with tab_bt:
    m = main_res["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR",       f"{m['cagr_%']:+.1f}%")
    c2.metric("Sharpe",     f"{m['sharpe']:.2f}")
    c3.metric("Max DD",     f"{m['max_drawdown_%']:.1f}%")
    c4.metric("Vol/ano",    f"{m['annual_vol_%']:.1f}%")
    c5.metric("Valor final",f"EUR {m['final_value']:,.0f}")

    eq  = main_res["equity"]
    fig = go.Figure(go.Scatter(
        x=eq.index, y=eq.round(2), mode="lines", name=strategy,
        hovertemplate="%{x|%Y-%m-%d}<br>EUR %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Equity curve", yaxis_title="Valor (EUR)",
                      hovermode="x unified", height=400)
    st.plotly_chart(fig, use_container_width=True)

    dd  = main_res["drawdown"] * 100
    fig2 = go.Figure(go.Scatter(
        x=dd.index, y=dd.round(2), fill="tozeroy", mode="lines",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    ))
    fig2.update_layout(title="Drawdown (%)", yaxis_title="%", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    wh   = main_res["weights_history"]
    fig3 = go.Figure()
    for ticker in wh.columns:
        w = wh[ticker] * 100
        if w.max() > 0.5:
            fig3.add_trace(go.Scatter(
                x=w.index, y=w.round(2), name=ticker,
                stackgroup="one", mode="lines",
            ))
    fig3.update_layout(title="Alocação ao longo do tempo (%)",
                       yaxis_range=[0, 100], height=350)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Log de rebalanceamentos"):
        st.dataframe(main_res["trades"], use_container_width=True)
        if not main_res["trades"].empty:
            st.caption(
                f"Fees totais: EUR {main_res['trades']['cost_eur'].sum():,.2f}"
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Análise técnica
# ════════════════════════════════════════════════════════════════════════════
with tab_ta:
    ticker_ta = st.selectbox("Ativo", selected)

    @st.cache_data(show_spinner=False)
    def load_ohlcv_cached(ticker, start, end):
        from src.data_loader import get_ohlcv
        return get_ohlcv(ticker, start=start, end=end)

    ohlcv = load_ohlcv_cached(ticker_ta, start_date, end_date)
    df_ta = all_indicators(ohlcv)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[
            f"{ticker_ta} — Candlestick + Médias + Bollinger",
            "MACD", "RSI (14)",
        ],
        vertical_spacing=0.05,
    )

    fig.add_trace(go.Candlestick(
        x=df_ta.index, open=df_ta.Open, high=df_ta.High,
        low=df_ta.Low, close=df_ta.Close, name="Preço",
    ), row=1, col=1)

    for col, color, dash in [
        ("SMA_50",  "blue",   "solid"),
        ("SMA_200", "orange", "solid"),
        ("bb_upper","gray",   "dot"),
        ("bb_lower","gray",   "dot"),
    ]:
        if col in df_ta.columns:
            fig.add_trace(go.Scatter(
                x=df_ta.index, y=df_ta[col], name=col,
                line=dict(color=color, width=1, dash=dash),
            ), row=1, col=1)

    # Suporte e resistência
    sr = support_resistance(df_ta["Close"])
    for lvl in sr["resistance"][:3]:
        fig.add_hline(y=lvl, line_dash="dash", line_color="red",
                      line_width=0.8, row=1, col=1)
    for lvl in sr["support"][:3]:
        fig.add_hline(y=lvl, line_dash="dash", line_color="green",
                      line_width=0.8, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_ta.index, y=df_ta["macd"],
        name="MACD", line=dict(color="blue", width=1),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_ta.index, y=df_ta["signal"],
        name="Signal", line=dict(color="orange", width=1),
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=df_ta.index, y=df_ta["histogram"],
        name="Histograma", marker_color="gray",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_ta.index, y=df_ta["RSI_14"],
        name="RSI", line=dict(color="purple", width=1),
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=800, xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk
# ════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("VaR e CVaR por perfil")

    var_rows = []
    for profile in ["low", "medium", "high"]:
        w   = optimize_for_profile(prices, profile)
        var = portfolio_var_cvar(w, returns, capital=capital)
        var_rows.append({
            "Perfil":          profile.upper(),
            "VaR diário %":    var.get("daily_var_%", 0),
            "VaR diário EUR":  var.get("daily_var_eur", 0),
            "CVaR diário %":   var.get("daily_cvar_%", 0),
            "CVaR diário EUR": var.get("daily_cvar_eur", 0),
            "VaR anual %":     var.get("annual_var_%", 0),
        })
    st.dataframe(pd.DataFrame(var_rows).set_index("Perfil"),
                 use_container_width=True)

    st.subheader("Stress tests")
    fig = go.Figure()
    colors = {"low": "#2E75B6", "medium": "#0F6E56", "high": "#854F0B"}
    for profile in ["low", "medium", "high"]:
        w      = optimize_for_profile(prices, profile)
        stress = run_stress_tests(w, capital=capital)
        fig.add_trace(go.Bar(
            name=profile.upper(),
            x=stress["cenario"],
            y=stress["perda_%"],
            marker_color=colors[profile],
            text=stress["perda_%"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", yaxis_title="Perda (%)",
        xaxis_tickangle=-15, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Alocação
# ════════════════════════════════════════════════════════════════════════════
with tab_alloc:
    split = {
        "low":    low_pct  / 100,
        "medium": med_pct  / 100,
        "high":   high_pct / 100,
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Low",    f"EUR {capital * split['low']:,.0f}",    f"{low_pct}%")
    c2.metric("Medium", f"EUR {capital * split['medium']:,.0f}", f"{med_pct}%")
    c3.metric("High",   f"EUR {capital * split['high']:,.0f}",   f"{high_pct}%")

    for profile in ["low", "medium", "high"]:
        cap_p  = capital * split[profile]
        if cap_p < 1:
            continue
        last_w = prof_res[profile]["weights_history"].iloc[-1]
        w_dict = last_w[last_w > 0.005].to_dict()

        c_pie, c_tbl = st.columns(2)
        with c_pie:
            fig = go.Figure(go.Pie(
                labels=list(w_dict.keys()),
                values=[v * 100 for v in w_dict.values()],
                textinfo="label+percent",
            ))
            fig.update_layout(
                title=f"{profile.upper()} — EUR {cap_p:,.0f}",
                height=300, margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        with c_tbl:
            rows = [
                {
                    "Ativo":     t,
                    "Peso %":    round(w * 100, 1),
                    "Capital EUR": round(cap_p * w, 0),
                }
                for t, w in sorted(w_dict.items(), key=lambda x: -x[1])
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Walk-forward
# ════════════════════════════════════════════════════════════════════════════
with tab_wf:
    st.caption(
        "Treina em 18 meses, testa nos 6 seguintes — "
        "avalia se a estratégia generaliza out-of-sample."
    )
    with st.spinner("A correr walk-forward..."):
        wf = walk_forward_backtest(
            prices, weight_fn,
            train_months=18, test_months=6,
            initial_capital=capital,
        )

    if "error" in wf:
        st.error(wf["error"])
    else:
        m = wf["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("OOS CAGR",   f"{m['cagr_%']:+.1f}%")
        c2.metric("OOS Sharpe", f"{m['sharpe']:.2f}")
        c3.metric("OOS Max DD", f"{m['max_drawdown_%']:.1f}%")

        eq  = wf["equity"]
        fig = go.Figure(go.Scatter(
            x=eq.index, y=eq.round(2), mode="lines",
        ))
        fig.update_layout(
            title="Equity walk-forward (só OOS)",
            yaxis_title="Valor (EUR)", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            wf["windows"], x="test_start", y="oos_sharpe",
            color="oos_sharpe", color_continuous_scale="RdYlGn",
            title="Sharpe por janela OOS",
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(wf["windows"], use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Explorar dados
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("Estatísticas por ativo")
    st.dataframe(summary_stats(prices), use_container_width=True)

    norm = prices / prices.iloc[0]
    fig  = px.line(norm, title="Retorno acumulado (base 1.0)")
    st.plotly_chart(fig, use_container_width=True)

    corr = returns.corr().round(2)
    fig2 = px.imshow(
        corr, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, text_auto=True,
        title="Correlação entre ativos",
    )
    st.plotly_chart(fig2, use_container_width=True)