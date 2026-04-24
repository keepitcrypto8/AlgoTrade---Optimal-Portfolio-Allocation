# src/data_loader.py
"""
Download e preparação de dados financeiros.

Conceitos importantes:
  - Preços ajustados: incorporam dividendos e splits — essencial para retornos correctos
  - Cache local: evita re-downloads, poupa tempo e respeita limites da API
  - Forward fill: alinha calendários bolsa (5 dias) com cripto (7 dias)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_prices(
    tickers: list,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Preços de fecho ajustados para uma lista de tickers.
    Columns = tickers, Index = DatetimeIndex (dias úteis).
    """
    import yfinance as yf

    key   = "_".join(sorted(tickers))
    cache = DATA_DIR / f"prices_{key}_{start[:4]}_{end[:4]}.csv"

    if use_cache and cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[cache] {df.shape[0]} dias x {df.shape[1]} ativos")
        return df

    print(f"[download] {len(tickers)} tickers de {start} a {end}...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    if len(tickers) == 1:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        df = raw["Close"]
        df = df[[t for t in tickers if t in df.columns]]

    df = df.dropna(how="all").ffill().dropna()
    df.to_csv(cache)
    print(f"[ok] {df.shape[0]} dias, {df.shape[1]} ativos")
    return df


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Retornos diários simples.
    retorno = (preco_hoje / preco_ontem) - 1
    """
    return prices.pct_change().dropna()


def get_ohlcv(
    ticker: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Dados OHLCV completos para um ticker.
    Colunas: Open, High, Low, Close, Volume — sempre índice simples.
    """
    import yfinance as yf

    cache = DATA_DIR / f"ohlcv_{ticker}_{start[:4]}_{end[:4]}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df

    print(f"[download] OHLCV {ticker}...")
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)

    # yfinance recente devolve MultiIndex — achatar para colunas simples
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(cache)
    return df


def summary_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas por ativo, ordenadas por Sharpe."""
    rets   = get_returns(prices)
    n_yrs  = len(prices) / 252
    cagr   = ((prices.iloc[-1] / prices.iloc[0]) ** (1 / n_yrs) - 1) * 100
    vol    = rets.std() * np.sqrt(252) * 100
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    max_dd = ((prices / prices.cummax()) - 1).min() * 100

    return pd.DataFrame({
        "cagr_%":    cagr,
        "vol_%":     vol,
        "sharpe":    sharpe,
        "max_dd_%":  max_dd,
    }).round(2).sort_values("sharpe", ascending=False)