# src/indicators.py
import pandas as pd
import numpy as np


def _to_series(data) -> pd.Series:
    """Garante que o input é sempre uma Series, mesmo que venha como DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def sma(close, window: int) -> pd.Series:
    s = _to_series(close)
    result = s.rolling(window).mean()
    result.name = f"SMA_{window}"
    return result


def ema(close, window: int) -> pd.Series:
    s = _to_series(close)
    result = s.ewm(span=window, adjust=False).mean()
    result.name = f"EMA_{window}"
    return result


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.
    Sinais: MACD cruza acima da signal line = momentum positivo.
    Histograma a crescer = tendência a ganhar força.
    """
    s          = _to_series(close)
    ema_fast   = s.ewm(span=fast,   adjust=False).mean()
    ema_slow   = s.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line

    return pd.DataFrame({
        "macd":      macd_line.values,
        "signal":    signal_line.values,
        "histogram": histogram.values,
    }, index=s.index)


def rsi(close, window: int = 14) -> pd.Series:
    """
    RSI — 0 a 100.
    >70 sobrecomprado, <30 sobrevendido.
    Não usar sozinho — em tendências fortes pode ficar extremo muito tempo.
    """
    s         = _to_series(close)
    delta     = s.diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(span=window, adjust=False).mean()
    avg_loss  = loss.ewm(span=window, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    result    = 100 - 100 / (1 + rs)
    result.name = f"RSI_{window}"
    return result


def bollinger_bands(close, window: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Bandas de Bollinger.
    Squeeze (bandas estreitas) precede movimento forte.
    Preço na banda inferior em tendência de alta = possível entrada.
    """
    s     = _to_series(close)
    mid   = s.rolling(window).mean()
    std   = s.rolling(window).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct_b = (s - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_middle": mid.values,
        "bb_upper":  upper.values,
        "bb_lower":  lower.values,
        "bb_pct_b":  pct_b.values,
    }, index=s.index)


def atr(high, low, close, window: int = 14) -> pd.Series:
    """
    ATR — Average True Range. Mede volatilidade em unidades de preço.
    Útil para stop-loss: stop = entrada - 2 * ATR.
    """
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)

    result = tr.ewm(span=window, adjust=False).mean()
    result.name = f"ATR_{window}"
    return result


def support_resistance(close, window: int = 20, min_touches: int = 2) -> dict:
    """
    Deteção automática de suporte e resistência por máximos/mínimos locais.
    Um nível é válido se o preço o tocou (dentro de 0.5%) pelo menos min_touches vezes.
    """
    s = _to_series(close)

    local_max = s[(s.shift(1) < s) & (s.shift(-1) < s)]
    local_min = s[(s.shift(1) > s) & (s.shift(-1) > s)]

    def cluster_levels(levels: pd.Series) -> list:
        if len(levels) == 0:
            return []
        sorted_l = sorted(levels.values)
        clusters, current = [], [sorted_l[0]]
        for lvl in sorted_l[1:]:
            if abs(lvl - current[-1]) / current[-1] < 0.005:
                current.append(lvl)
            else:
                clusters.append(round(float(np.mean(current)), 4))
                current = [lvl]
        clusters.append(round(float(np.mean(current)), 4))
        return clusters

    def count_touches(levels: list) -> list:
        valid = []
        for lvl in levels:
            touches = ((s - lvl).abs() / lvl < 0.005).sum()
            if touches >= min_touches:
                valid.append(lvl)
        return valid

    return {
        "resistance": count_touches(cluster_levels(local_max)),
        "support":    count_touches(cluster_levels(local_min)),
    }


def fibonacci_levels(swing_high: float, swing_low: float) -> dict:
    """
    Níveis de retração de Fibonacci.
    Os níveis 0.382, 0.500 e 0.618 são os mais usados como suporte/resistência dinâmica.
    """
    diff   = swing_high - swing_low
    ratios = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
    return {f"fib_{r:.3f}": round(swing_high - diff * r, 4) for r in ratios}


def all_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os indicadores para um DataFrame OHLCV.
    Devolve um DataFrame único com todas as colunas.
    """
    # Extrair colunas — compatível com MultiIndex do yfinance recente
    def get_col(df, name):
        if isinstance(df.columns, pd.MultiIndex):
            cols = [c for c in df.columns if c[0] == name]
            return df[cols[0]] if cols else pd.Series(dtype=float)
        return df[name] if name in df.columns else pd.Series(dtype=float)

    close = get_col(ohlcv, "Close")
    high  = get_col(ohlcv, "High")
    low   = get_col(ohlcv, "Low")

    # Base OHLCV normalizada (sem MultiIndex)
    base = pd.DataFrame({
        "Open":   get_col(ohlcv, "Open").values,
        "High":   high.values,
        "Low":    low.values,
        "Close":  close.values,
        "Volume": get_col(ohlcv, "Volume").values,
    }, index=close.index)

    indicators = pd.concat([
        sma(close, 20),
        sma(close, 50),
        sma(close, 200),
        ema(close, 12),
        ema(close, 26),
        macd(close),
        rsi(close, 14),
        bollinger_bands(close),
        atr(high, low, close, 14),
    ], axis=1)

    return pd.concat([base, indicators], axis=1)