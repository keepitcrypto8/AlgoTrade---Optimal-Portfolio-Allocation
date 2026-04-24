# src/backtester.py
"""
Motor de backtesting com rebalanceamento contínuo.

Suporta:
  - Rebalanceamento periódico: diário, semanal, mensal, trimestral
  - Drift-triggered: rebalanceia quando qualquer peso desvia > threshold
  - Walk-forward validation: treina em janelas rolantes, testa out-of-sample
  - Custos de transacção realistas em cada rebalanceamento

Princípio fundamental:
  Em cada rebalanceamento, o weight_fn recebe APENAS dados históricos
  até essa data — nunca informação do futuro.
"""

import numpy as np
import pandas as pd
from typing import Callable


def run_backtest(
    prices: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], dict],
    initial_capital: float = 10_000.0,
    fees: float = 0.001,
    rebalance_freq: str = "monthly",
    drift_threshold: float = 0.05,
    lookback_days: int = 252,
    min_history: int = 60,
) -> dict:
    """
    Simula o portfolio dia a dia.

    Parâmetros:
      prices          : preços diários (colunas = tickers)
      weight_fn       : função que recebe prices_slice e devolve {ticker: weight}
                        É chamada em cada rebalanceamento com dados até hoje.
      initial_capital : capital inicial em euros
      fees            : custo por transacção (0.001 = 0.1%)
      rebalance_freq  : 'daily' | 'weekly' | 'monthly' | 'quarterly'
      drift_threshold : rebalanceia se qualquer peso deriva > este valor
      lookback_days   : janela de história passada ao weight_fn
      min_history     : dias mínimos antes do primeiro rebalanceamento

    Retorna dict com:
      equity          : pd.Series — valor do portfolio em cada dia
      weights_history : pd.DataFrame — pesos reais em cada dia
      returns         : pd.Series — retornos diários
      drawdown        : pd.Series — drawdown em cada dia
      metrics         : dict — CAGR, Sharpe, MaxDD, etc.
      trades          : pd.DataFrame — log de todos os rebalanceamentos
    """
    tickers      = list(prices.columns)
    dates        = prices.index
    holdings     = pd.Series(0.0, index=tickers)
    cash         = initial_capital

    equity_curve = pd.Series(0.0, index=dates)
    weights_hist = pd.DataFrame(0.0, index=dates, columns=tickers)
    rebalance_log= []
    current_w    = pd.Series(1.0 / len(tickers), index=tickers)
    reb_dates    = _rebalance_dates(dates, rebalance_freq)

    for i, date in enumerate(dates):
        px         = prices.loc[date]
        port_value = cash + (holdings * px).sum()

        # ── Decidir se rebalanceia hoje ───────────────────────────────────────
        scheduled  = (date in reb_dates and i >= min_history)
        first_reb  = (i == min_history)
        should_reb = scheduled or first_reb

        if should_reb:
            # Passa apenas dados históricos até hoje (sem lookahead)
            hist = prices.iloc[max(0, i - lookback_days): i]

            if len(hist) >= 20:
                try:
                    new_w_dict = weight_fn(hist)
                    target = pd.Series(0.0, index=tickers)
                    for t, w in new_w_dict.items():
                        if t in target.index:
                            target[t] = w
                    if target.sum() > 0:
                        target /= target.sum()
                    else:
                        target = current_w.copy()
                except Exception:
                    target = current_w.copy()
            else:
                target = current_w.copy()

            if port_value > 1.0:
                port_value = _execute_rebalance(
                    holdings, target, px, port_value, fees,
                    rebalance_log, date
                )
                current_w = target.copy()

        # ── Drift check entre rebalanceamentos agendados ──────────────────────
        elif i > 0 and drift_threshold < 1.0 and port_value > 1.0:
            actual_w = (holdings * px) / port_value
            max_drift = (actual_w - current_w).abs().max()

            if max_drift > drift_threshold:
                port_value = _execute_rebalance(
                    holdings, current_w, px, port_value, fees,
                    rebalance_log, date
                )

        # ── Registar estado de fim de dia ─────────────────────────────────────
        equity_curve[date] = cash + (holdings * px).sum()
        if equity_curve[date] > 1e-9:
            weights_hist.loc[date] = (holdings * px) / equity_curve[date]

    # ── Métricas finais ───────────────────────────────────────────────────────
    equity_curve = equity_curve.replace(0.0, np.nan).ffill().fillna(initial_capital)
    daily_rets   = equity_curve.pct_change().dropna()
    drawdown     = (equity_curve / equity_curve.cummax()) - 1
    metrics      = _compute_metrics(equity_curve, daily_rets, initial_capital)
    trades_df    = (pd.DataFrame(rebalance_log)
                    if rebalance_log
                    else pd.DataFrame(columns=["date", "port_value", "cost_eur"]))

    return {
        "equity":          equity_curve,
        "weights_history": weights_hist,
        "returns":         daily_rets,
        "drawdown":        drawdown,
        "metrics":         metrics,
        "trades":          trades_df,
    }


def _execute_rebalance(
    holdings: pd.Series,
    target_w: pd.Series,
    prices: pd.Series,
    port_value: float,
    fees: float,
    log: list,
    date,
) -> float:
    """
    Executa o rebalanceamento: calcula o delta de acções a comprar/vender,
    deduz as fees, e actualiza holdings in-place.
    Retorna o novo valor do portfolio após custos.
    """
    target_value  = target_w * port_value
    target_shares = target_value / prices.replace(0, np.nan).fillna(1)
    delta_shares  = target_shares - holdings

    traded_value  = (delta_shares.abs() * prices).sum()
    cost          = traded_value * fees

    holdings[:] = target_shares.values
    port_value -= cost

    log.append({
        "date":         date,
        "port_value":   round(float(port_value), 2),
        "traded_value": round(float(traded_value), 2),
        "cost_eur":     round(float(cost), 2),
    })
    return port_value


def _compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    initial: float,
    rf: float = 0.02,
) -> dict:
    """Calcula todas as métricas de avaliação do backtest."""
    n_years  = len(equity) / 252
    cagr     = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    vol      = returns.std() * np.sqrt(252)
    downside = returns[returns < 0].std() * np.sqrt(252)
    max_dd   = ((equity / equity.cummax()) - 1).min()
    sharpe   = (cagr - rf) / vol      if vol      > 0 else 0.0
    sortino  = (cagr - rf) / downside if downside > 0 else 0.0
    calmar   = cagr / abs(max_dd)     if max_dd   < 0 else 0.0

    return {
        "initial_capital":  round(float(initial), 2),
        "final_value":      round(float(equity.iloc[-1]), 2),
        "total_return_%":   round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "cagr_%":           round(cagr * 100, 2),
        "annual_vol_%":     round(vol * 100, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown_%":   round(max_dd * 100, 2),
        "calmar":           round(calmar, 3),
    }


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set:
    """Calcula o conjunto de datas de rebalanceamento agendado."""
    s = pd.Series(dates, index=dates)
    if freq == "daily":
        return set(dates)
    if freq == "weekly":
        return set(s.groupby(s.dt.isocalendar().week).first())
    if freq == "monthly":
        return set(s.groupby([s.dt.year, s.dt.month]).first())
    if freq == "quarterly":
        q = (s.dt.month - 1) // 3
        return set(s.groupby([s.dt.year, q]).first())
    raise ValueError(f"freq inválida: {freq}. Usar daily/weekly/monthly/quarterly")


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward_backtest(
    prices: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], dict],
    train_months: int = 18,
    test_months: int = 6,
    **backtest_kwargs,
) -> dict:
    """
    Walk-forward validation — a única forma honesta de avaliar uma estratégia.

    Em vez de testar em todo o histórico de uma vez (que permite overfitting),
    divide os dados em janelas rolantes:

      Janela 1: treina em meses 1-18  → testa em meses 19-24
      Janela 2: treina em meses 7-24  → testa em meses 25-30
      Janela 3: treina em meses 13-30 → testa em meses 31-36
      ...

    Os pesos são fixados no período de treino e aplicados ao período de teste
    sem qualquer ajuste — simulando o que aconteceria em produção.

    Retorna:
      equity  : curva de equity encadeada de todos os períodos OOS
      windows : DataFrame com métricas por janela
      metrics : métricas agregadas de todo o período OOS
    """
    all_equity  = []
    all_windows = []
    start_idx   = 0
    dates       = prices.index

    while True:
        train_end = start_idx + train_months * 21
        test_end  = train_end + test_months * 21

        if test_end >= len(dates):
            break

        train_prices = prices.iloc[start_idx:train_end]
        test_prices  = prices.iloc[train_end:test_end]

        if len(train_prices) < 60:
            break

        # Calcular pesos óptimos no período de treino
        try:
            fixed_w = weight_fn(train_prices)
        except Exception:
            start_idx += test_months * 21
            continue

        # Testar com pesos fixos — sem re-optimização dentro da janela de teste
        result = run_backtest(
            test_prices,
            weight_fn=lambda p, w=fixed_w: w,
            rebalance_freq="monthly",
            drift_threshold=0.99,   # sem drift rebalancing no teste
            min_history=0,
            **backtest_kwargs,
        )

        all_equity.append(result["equity"])
        all_windows.append({
            "train_start": dates[start_idx].date(),
            "train_end":   dates[train_end - 1].date(),
            "test_start":  dates[train_end].date(),
            "test_end":    dates[test_end - 1].date(),
            "oos_sharpe":  result["metrics"]["sharpe"],
            "oos_cagr_%":  result["metrics"]["cagr_%"],
            "oos_maxdd_%": result["metrics"]["max_drawdown_%"],
        })

        start_idx += test_months * 21

    if not all_equity:
        return {"error": "Dados insuficientes para walk-forward validation"}

    # Encadear curvas de equity (cada janela começa onde a anterior terminou)
    chained = all_equity[0].copy()
    for seg in all_equity[1:]:
        scale   = chained.iloc[-1] / seg.iloc[0]
        chained = pd.concat([chained, seg * scale])

    cr = chained.pct_change().dropna()
    return {
        "equity":  chained,
        "windows": pd.DataFrame(all_windows),
        "metrics": _compute_metrics(chained, cr, float(chained.iloc[0])),
    }