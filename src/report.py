# src/report.py
"""
Comparação de estratégias e exportação de resultados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compare_strategies(results: dict) -> pd.DataFrame:
    """
    Tabela de comparação de múltiplos backtests.
    results = {'nome': backtest_result_dict, ...}
    """
    rows = []
    for name, res in results.items():
        row = dict(res["metrics"])
        row["strategy"] = name
        t = res.get("trades", pd.DataFrame())
        if not t.empty:
            row["fees_totais_eur"] = round(t["cost_eur"].sum(), 2)
            row["n_rebalances"]    = len(t)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")
    order = ["cagr_%", "annual_vol_%", "sharpe", "sortino",
             "max_drawdown_%", "calmar", "final_value",
             "n_rebalances", "fees_totais_eur"]
    return df[[c for c in order if c in df.columns]].round(3)


def save_results(result: dict, label: str = "backtest") -> None:
    """Guarda equity curve e métricas em CSV."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"{label.replace(' ', '_')}_{ts}"

    result["equity"].to_csv(
        RESULTS_DIR / f"{stem}_equity.csv", header=["value_eur"]
    )
    pd.Series(result["metrics"]).to_csv(
        RESULTS_DIR / f"{stem}_metrics.csv", header=["value"]
    )
    print(f"[saved] {stem}_*.csv em results/")