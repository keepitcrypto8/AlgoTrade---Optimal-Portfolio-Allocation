# src/config.py
"""
Configuração central do projecto AlgoTrade.
Edita este ficheiro para mudar ativos, datas e perfis de risco.
Todos os outros módulos importam daqui.
"""

# ── Universo de ativos ────────────────────────────────────────────────────────
ASSETS = {
    "equities": [
        "SPY",    # S&P 500 ETF — 500 maiores empresas dos EUA
        "QQQ",    # Nasdaq 100 ETF — tecnologia e crescimento
        "AAPL",   # Apple — exemplo de acção individual
    ],
    "etfs": [
        "GLD",    # Gold ETF — protecção contra inflação
        "TLT",    # Treasury Bond ETF — bonds longo prazo
    ],
    "crypto": [
        "BTC-USD",
        "ETH-USD",
    ],
    "commodities": [
        "USO",    # US Oil Fund
    ],
}

ALL_ASSETS = [ticker for group in ASSETS.values() for ticker in group]
ASSET_CATEGORIES = list(ASSETS.keys())

# ── Datas ─────────────────────────────────────────────────────────────────────
BACKTEST_START = "2019-01-01"
BACKTEST_END   = "2024-12-31"

# ── Custos de transacção ──────────────────────────────────────────────────────
FEES     = 0.001
SLIPPAGE = 0.001

# ── Capital inicial ────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000

# ── Perfis de risco ───────────────────────────────────────────────────────────
RISK_PROFILES = {
    "low": {
        "max_weight_crypto": 0.00,
        "max_vol":           0.08,
        "max_dd":            0.10,
        "description":       "Conservador — bonds + ouro + equity defensivo",
    },
    "medium": {
        "max_weight_crypto": 0.10,
        "max_vol":           0.15,
        "max_dd":            0.20,
        "description":       "Moderado — equities globais + alguma cripto",
    },
    "high": {
        "max_weight_crypto": 0.30,
        "max_vol":           0.28,
        "max_dd":            0.40,
        "description":       "Agressivo — alto crescimento + cripto",
    },
}

# ── Rebalanceamento ───────────────────────────────────────────────────────────
REBALANCE_FREQ  = "monthly"
DRIFT_THRESHOLD = 0.05
LOOKBACK_DAYS   = 252