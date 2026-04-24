# src/data_loader.py
"""
Módulo de download e preparação de dados financeiros.

CONCEITOS IMPORTANTES:
  - Preços ajustados: o yfinance devolve preços que já incorporam dividendos
    e splits. Isto é ESSENCIAL — sem ajuste, os retornos calculados são errados.
  - Cache local: evitamos re-downloads repetidos guardando os dados em CSV.
    Poupa tempo e evita bater nos limites da API.
  - Forward fill: bolsas fecham ao fim-de-semana, cripto não. O ffill() preenche
    os dias sem cotação com o último preço conhecido, alinhando os calendários.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Pasta para guardar os dados em cache
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_prices(
    tickers: list,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    use_cache: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Descarrega preços de fecho ajustados para uma lista de tickers.

    Parâmetros:
      tickers       : lista de símbolos, ex: ["SPY", "BTC-USD", "GLD"]
      start / end   : datas no formato "YYYY-MM-DD"
      use_cache     : se True, usa CSV local se existir (mais rápido)
      force_download: ignora o cache e faz download fresco

    Retorna:
      DataFrame com datas no índice e tickers nas colunas.
      Cada célula = preço de fecho ajustado nesse dia.
    """
    import yfinance as yf

    # Chave única para o cache baseada nos tickers e datas
    key   = "_".join(sorted(tickers))
    cache = DATA_DIR / f"prices_{key}_{start[:4]}_{end[:4]}.csv"

    if use_cache and cache.exists() and not force_download:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[cache] {df.shape[0]} dias × {df.shape[1]} ativos")
        return df

    print(f"[download] A descarregar {len(tickers)} tickers de {start} a {end}…")

    # yfinance aceita uma lista de tickers ou um único string
    # auto_adjust=True → preços já ajustados de dividendos e splits
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Quando tens 1 ticker, yfinance devolve todas as colunas OHLCV para esse ticker
    # Quando tens vários, devolve um MultiIndex com (coluna, ticker)
    if len(tickers) == 1:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        df = raw["Close"]
        # Garantir que as colunas ficam na ordem da lista original
        df = df[[t for t in tickers if t in df.columns]]

    # ── Limpeza ──────────────────────────────────────────────────────────────
    # Remove dias onde TODOS os ativos têm NaN (ex: feriados globais)
    df = df.dropna(how="all")

    # Forward fill: preenche dias sem cotação com o último preço
    # (alinha calendários: cripto=7 dias/semana, bolsas=5 dias/semana)
    df = df.ffill()

    # Remove qualquer NaN restante (primeiras linhas antes do IPO de algum ativo)
    df = df.dropna()

    # Guardar em cache para uso futuro
    df.to_csv(cache)
    print(f"[ok] {df.shape[0]} dias, {df.shape[1]} ativos, 0 NaN")

    return df


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos diários simples a partir de preços de fecho.

    Retorno diário = (preço_hoje / preço_ontem) - 1
    Ex: SPY passou de 400 para 404 → retorno = (404/400) - 1 = 0.01 = +1%

    Nota: usamos retornos simples (não log-retornos) porque são mais intuitivos
    e funcionam melhor para portfolios multi-asset com pesos dinâmicos.
    """
    return prices.pct_change().dropna()


def get_category_prices(
    category: str,
    start: str = "2019-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Descarrega preços apenas para uma categoria específica (ex: 'crypto').
    Útil para análise focada numa classe de activos.
    """
    from src.config import ASSETS
    tickers = ASSETS.get(category, [])
    if not tickers:
        raise ValueError(f"Categoria '{category}' não existe em config.ASSETS")
    return get_prices(tickers, start=start, end=end)


def summary_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela de estatísticas descritivas por ativo.

    Métricas calculadas:
      cagr_%      : Compound Annual Growth Rate — retorno anual composto
      vol_%       : Volatilidade anualizada (desvio padrão × sqrt(252))
      sharpe      : Sharpe ratio aproximado (CAGR / vol, sem taxa livre de risco)
      max_dd_%    : Maximum Drawdown — maior queda do pico ao vale
      total_ret_% : Retorno total no período
    """
    rets   = get_returns(prices)
    n_years = len(prices) / 252   # número de anos no período

    cagr     = ((prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1) * 100
    vol      = rets.std() * np.sqrt(252) * 100
    sharpe   = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    max_dd   = ((prices / prices.cummax()) - 1).min() * 100
    total_r  = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100

    stats = pd.DataFrame({
        "total_ret_%": total_r,
        "cagr_%":      cagr,
        "vol_%":       vol,
        "sharpe":      sharpe,
        "max_dd_%":    max_dd,
    }).round(2)

    # Ordenar por Sharpe (melhor retorno ajustado ao risco primeiro)
    return stats.sort_values("sharpe", ascending=False)