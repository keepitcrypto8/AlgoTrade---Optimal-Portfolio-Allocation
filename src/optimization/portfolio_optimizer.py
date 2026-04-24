# src/optimization/portfolio_optimizer.py
"""
5 métodos de optimização de portfolio.

Todos os métodos recebem um DataFrame de preços ou retornos
e devolvem um dict {ticker: weight} que soma exactamente 1.0.
Nenhum requer bibliotecas externas além de scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ── Helpers internos ───────────────────────────────────────────────────────────

def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos diários a partir de preços."""
    return prices.pct_change().dropna()


def _ann_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    Covariância anualizada com Ledoit-Wolf shrinkage se sklearn disponível,
    sample covariance como fallback.

    Ledoit-Wolf é preferível porque a sample covariance com poucos dados
    é instável — pequenas mudanças nos preços alteram muito a matriz.
    """
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(returns.values).covariance_ * 252
    except Exception:
        return returns.cov().values * 252


def _ann_mean(returns: pd.DataFrame) -> np.ndarray:
    """Retornos esperados anualizados (média histórica × 252)."""
    return returns.mean().values * 252


def portfolio_stats(weights: np.ndarray, mu: np.ndarray,
                    cov: np.ndarray, rf: float = 0.02) -> dict:
    """Estatísticas de um portfolio com dados pesos."""
    w   = np.array(weights)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sr  = (ret - rf) / vol if vol > 1e-9 else 0.0
    return {"return": round(ret, 4), "volatility": round(vol, 4),
            "sharpe": round(sr, 4)}


# ── Método 1: Equal Weight ─────────────────────────────────────────────────────

def equal_weight(tickers: list) -> dict:
    """
    Cada ativo recebe 1/N.
    O benchmark — surpreendentemente difícil de bater de forma consistente.
    """
    n = len(tickers)
    return {t: round(1.0 / n, 6) for t in tickers}


# ── Método 2: Inverse Volatility ──────────────────────────────────────────────

def inverse_vol(returns: pd.DataFrame) -> dict:
    """
    Peso = 1/volatilidade, normalizado para somar 1.
    Risk parity simples — cada ativo contribui igual para o risco total.
    """
    vol = returns.std()
    inv = 1.0 / vol.replace(0, np.nan).fillna(1.0)
    w   = inv / inv.sum()
    return {t: round(float(w[t]), 6) for t in returns.columns}


# ── Método 3: Max Sharpe ───────────────────────────────────────────────────────

def max_sharpe(returns: pd.DataFrame, rf: float = 0.02,
               max_weight: float = 0.40) -> dict:
    """
    Maximiza (retorno - rf) / volatilidade.
    max_weight: limite por ativo para evitar concentração (default 40%).

    Aviso: muito sensível a estimativas de retorno. Usar lookback >= 252 dias.
    """
    mu  = _ann_mean(returns)
    cov = _ann_cov(returns)
    n   = len(returns.columns)
    w0  = np.ones(n) / n

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 1e-9 else 0.0

    res = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not res.success:
        return inverse_vol(returns)  # fallback

    w = np.clip(res.x, 0, max_weight)
    w /= w.sum()
    return {t: round(float(w[i]), 6) for i, t in enumerate(returns.columns)}


# ── Método 4: Min Volatility ───────────────────────────────────────────────────

def min_volatility(returns: pd.DataFrame,
                   max_weight: float = 0.40) -> dict:
    """
    Minimiza a volatilidade do portfolio.
    Ideal para perfil LOW — maximiza diversificação, ignora retorno esperado.
    """
    cov = _ann_cov(returns)
    n   = len(returns.columns)
    w0  = np.ones(n) / n

    res = minimize(
        lambda w: np.sqrt(w @ cov @ w),
        w0, method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not res.success:
        return inverse_vol(returns)

    w = np.clip(res.x, 0, max_weight)
    w /= w.sum()
    return {t: round(float(w[i]), 6) for i, t in enumerate(returns.columns)}


# ── Método 5: HRP ──────────────────────────────────────────────────────────────

def hrp(returns: pd.DataFrame) -> dict:
    """
    Hierarchical Risk Parity (López de Prado, 2016).

    Passos:
      1. Calcula matriz de correlação e converte em distâncias
      2. Clustering hierárquico dos ativos por similaridade
      3. Reordena a matriz de covariância (quasi-diagonal)
      4. Bissecção recursiva: divide em dois clusters, distribui
         o risco proporcionalmente à variância de cada cluster

    Vantagem: não inverte a matriz de covariância — mais estável.
    """
    corr = returns.corr().values
    cov  = _ann_cov(returns)
    tickers = list(returns.columns)
    n = len(tickers)

    # Passo 1: distâncias
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, 1))
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2

    # Passo 2: clustering
    link = linkage(squareform(dist, checks=False), method="single")

    # Passo 3: reordenação quasi-diagonal
    sort_ix = _quasi_diag(link, n)
    sorted_tickers = [tickers[i] for i in sort_ix]

    # Passo 4: bissecção recursiva
    weights  = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]
    cov_df   = pd.DataFrame(cov, index=tickers, columns=tickers)

    while clusters:
        cl = clusters.pop(0)
        if len(cl) <= 1:
            continue
        mid = len(cl) // 2
        left, right = cl[:mid], cl[mid:]

        def cluster_var(tks):
            sub = cov_df.loc[tks, tks].values
            w   = np.ones(len(tks)) / len(tks)
            return float(w @ sub @ w)

        var_l = cluster_var(left)
        var_r = cluster_var(right)
        alpha = 1 - var_l / (var_l + var_r)

        weights[left]  *= alpha
        weights[right] *= (1 - alpha)

        if len(left)  > 1: clusters.append(left)
        if len(right) > 1: clusters.append(right)

    weights /= weights.sum()
    return {t: round(float(weights[t]), 6) for t in tickers}


def _quasi_diag(link: np.ndarray, n: int) -> list:
    """Extrai a ordem quasi-diagonal das folhas do dendrograma."""
    link    = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    while sort_ix.max() >= n:
        sort_ix.index = range(0, len(sort_ix) * 2, 2)
        df0   = sort_ix[sort_ix >= n]
        i, j  = df0.index, df0.values - n
        sort_ix[i] = link[j, 0]
        new   = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, new]).sort_index()
        sort_ix.index = range(len(sort_ix))

    return sort_ix.tolist()


# ── Perfis de risco ────────────────────────────────────────────────────────────

def optimize_for_profile(
    prices: pd.DataFrame,
    profile: str = "medium",
    lookback_days: int = 252,
    crypto_tickers: list = None,
) -> dict:
    """
    Escolhe o método de optimização adequado ao perfil de risco
    e aplica os constraints definidos em config.py.

      low    → min_volatility, sem cripto
      medium → max_sharpe, cripto limitada a 10%
      high   → hrp, cripto limitada a 30%

    Retorna sempre um dict que soma 1.0.
    """
    from src.config import RISK_PROFILES, ASSETS

    if crypto_tickers is None:
        crypto_tickers = ASSETS.get("crypto", [])

    cfg  = RISK_PROFILES[profile]
    rets = prices.iloc[-lookback_days:].pct_change().dropna()

    # ── Calcular pesos base ────────────────────────────────────────────────────
    if profile == "low":
        safe = [t for t in prices.columns if t not in crypto_tickers]
        if not safe:
            safe = list(prices.columns)
        w = min_volatility(rets[safe], max_weight=0.45)
        # Garantir que cripto fica a zero
        weights = {t: 0.0 for t in prices.columns}
        weights.update(w)

    elif profile == "medium":
        weights = max_sharpe(rets, max_weight=0.35)
        _apply_crypto_cap(weights, crypto_tickers, cfg["max_weight_crypto"])

    else:  # high
        weights = hrp(rets)
        _apply_crypto_cap(weights, crypto_tickers, cfg["max_weight_crypto"])

    # ── Limpar pesos pequenos e renormalizar ───────────────────────────────────
    weights = {t: v for t, v in weights.items() if v >= 0.005}
    total   = sum(weights.values())
    return {t: round(v / total, 6) for t, v in weights.items()}


def _apply_crypto_cap(weights: dict, crypto_tickers: list,
                      max_crypto: float) -> None:
    """In-place: reduz cripto para max_crypto e redistribui para non-crypto."""
    crypto_w = sum(weights.get(t, 0) for t in crypto_tickers)
    if crypto_w <= max_crypto or crypto_w == 0:
        return

    scale = max_crypto / crypto_w
    freed = 0.0
    for t in crypto_tickers:
        old = weights.get(t, 0)
        weights[t] = old * scale
        freed += old - weights[t]

    non_crypto   = [t for t in weights if t not in crypto_tickers]
    total_non_c  = sum(weights[t] for t in non_crypto) or 1.0
    for t in non_crypto:
        weights[t] += freed * (weights[t] / total_non_c)


# ── Fronteira eficiente ────────────────────────────────────────────────────────

def efficient_frontier(
    prices: pd.DataFrame,
    n_portfolios: int = 3000,
    rf: float = 0.02,
) -> pd.DataFrame:
    """
    Simula n_portfolios aleatórios e calcula retorno, volatilidade e Sharpe.
    Usar para plotar a fronteira eficiente no notebook.

    Retorna DataFrame com colunas: return, volatility, sharpe, + peso de cada ativo.
    """
    rets = prices.pct_change().dropna()
    mu   = _ann_mean(rets)
    cov  = _ann_cov(rets)
    n    = len(prices.columns)

    rows = []
    for _ in range(n_portfolios):
        w   = np.random.dirichlet(np.ones(n))
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        sr  = (ret - rf) / vol if vol > 1e-9 else 0.0
        row = {"return": round(ret, 4), "volatility": round(vol, 4),
               "sharpe": round(sr, 4)}
        row.update({t: round(float(w[i]), 4)
                    for i, t in enumerate(prices.columns)})
        rows.append(row)

    return pd.DataFrame(rows)