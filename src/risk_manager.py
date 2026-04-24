# src/risk_manager.py
"""
Risk management — VaR, CVaR, stress tests, alocação de capital.

Conceitos:
  VaR (Value at Risk):
    "Com 95% de confiança, não perco mais de X num único dia."
    Limitação: não diz nada sobre o tamanho das perdas nos 5% piores dias.

  CVaR (Conditional VaR / Expected Shortfall):
    "Nos 5% piores dias, perco em média X."
    Mais informativo que VaR — mede a severidade do tail risk.

  Stress test:
    Aplica choques históricos conhecidos (COVID, 2022, GFC) ao portfolio
    actual e calcula a perda estimada. Não é probabilístico — é um cenário.
"""

import numpy as np
import pandas as pd


# ── VaR e CVaR ────────────────────────────────────────────────────────────────

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    VaR histórico — percentil dos retornos negativos.
    Retorna o valor positivo da perda (ex: 0.023 = 2.3% de perda).
    """
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    CVaR histórico — média das perdas além do VaR.
    Sempre >= VaR. Quanto maior a diferença VaR/CVaR, mais pesada a cauda.
    """
    var  = historical_var(returns, confidence)
    tail = returns[returns < -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def portfolio_var_cvar(
    weights: dict,
    returns: pd.DataFrame,
    confidence: float = 0.95,
    capital: float = 10_000,
) -> dict:
    """
    Calcula VaR e CVaR para o portfolio combinado.

    Retorna dict com valores em % e em euros para o capital dado.
    """
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights.get(t, 0.0) for t in tickers])
    if w.sum() == 0:
        return {}
    w /= w.sum()

    port_rets = (returns[tickers].fillna(0) * w).sum(axis=1)

    var_pct  = historical_var(port_rets, confidence)
    cvar_pct = historical_cvar(port_rets, confidence)

    return {
        "confidence":     confidence,
        "daily_var_%":    round(var_pct * 100, 3),
        "daily_var_eur":  round(var_pct * capital, 2),
        "daily_cvar_%":   round(cvar_pct * 100, 3),
        "daily_cvar_eur": round(cvar_pct * capital, 2),
        "annual_var_%":   round(var_pct * np.sqrt(252) * 100, 2),
    }


# ── Stress tests ──────────────────────────────────────────────────────────────

STRESS_SCENARIOS = {
    "COVID crash (Fev-Mar 2020)": {
        "description": "Equities globais -34% em 5 semanas",
        "shocks": {
            "SPY": -0.34, "QQQ": -0.30, "AAPL": -0.30,
            "GLD":  0.05, "TLT":  0.10,
            "BTC-USD": -0.50, "ETH-USD": -0.60,
            "USO": -0.65,
        },
    },
    "Rate hike shock (2022)": {
        "description": "Fed sobe taxas — bonds e growth caem",
        "shocks": {
            "SPY": -0.20, "QQQ": -0.33, "AAPL": -0.28,
            "GLD":  0.00, "TLT": -0.30,
            "BTC-USD": -0.65, "ETH-USD": -0.68,
            "USO":  0.10,
        },
    },
    "Crypto winter (2022)": {
        "description": "Cripto perde 70-80% do valor",
        "shocks": {
            "SPY": -0.10, "QQQ": -0.15, "AAPL": -0.10,
            "GLD":  0.02, "TLT": -0.05,
            "BTC-USD": -0.75, "ETH-USD": -0.80,
            "USO":  0.00,
        },
    },
    "GFC-style (2008)": {
        "description": "Crash sistémico — equities -50% em 18 meses",
        "shocks": {
            "SPY": -0.50, "QQQ": -0.48, "AAPL": -0.55,
            "GLD":  0.25, "TLT":  0.30,
            "BTC-USD": -0.80, "ETH-USD": -0.85,
            "USO": -0.55,
        },
    },
    "Correção normal (10-15%)": {
        "description": "Pullback comum — acontece ~1x por ano",
        "shocks": {
            "SPY": -0.12, "QQQ": -0.15, "AAPL": -0.15,
            "GLD":  0.03, "TLT":  0.05,
            "BTC-USD": -0.25, "ETH-USD": -0.30,
            "USO": -0.10,
        },
    },
}


def run_stress_tests(
    weights: dict,
    capital: float = 10_000,
    scenarios: dict = None,
) -> pd.DataFrame:
    """
    Aplica cenários de stress ao portfolio actual.

    Retorna DataFrame com perda estimada por cenário,
    ordenado do pior para o melhor.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    rows = []
    for name, sc in scenarios.items():
        pnl_pct = sum(
            weights.get(t, 0) * sc["shocks"].get(t, 0)
            for t in weights
        ) * 100

        rows.append({
            "cenario":          name,
            "descricao":        sc["description"],
            "perda_%":          round(pnl_pct, 1),
            "perda_eur":        round(pnl_pct / 100 * capital, 0),
            "valor_restante_eur": round(capital + pnl_pct / 100 * capital, 0),
        })

    return pd.DataFrame(rows).sort_values("perda_%")


# ── Alocação de capital entre perfis ──────────────────────────────────────────

def risk_budget_allocation(
    total_capital: float,
    investor_split: dict,
    profile_metrics: dict,
) -> pd.DataFrame:
    """
    Tabela de alocação do capital total pelos 3 perfis.

    investor_split  : {'low': 0.4, 'medium': 0.4, 'high': 0.2}
    profile_metrics : {'low': metrics_dict, 'medium': ..., 'high': ...}
    """
    rows = []
    for profile, pct in investor_split.items():
        cap = total_capital * pct
        m   = profile_metrics.get(profile, {})
        rows.append({
            "perfil":          profile.upper(),
            "alocacao_%":      round(pct * 100, 1),
            "capital_eur":     round(cap, 0),
            "cagr_esperado_%": m.get("cagr_%", 0),
            "vol_anual_%":     m.get("annual_vol_%", 0),
            "max_drawdown_%":  m.get("max_drawdown_%", 0),
            "sharpe":          m.get("sharpe", 0),
        })

    return pd.DataFrame(rows)