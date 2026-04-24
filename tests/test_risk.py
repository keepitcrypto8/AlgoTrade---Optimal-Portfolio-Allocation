# tests/test_risk.py
import sys
sys.path.insert(0, ".")

from src.data_loader import get_prices, get_returns
from src.backtester import run_backtest
from src.optimization.portfolio_optimizer import optimize_for_profile
from src.risk_manager import portfolio_var_cvar, run_stress_tests, risk_budget_allocation
from src.config import ALL_ASSETS

prices  = get_prices(ALL_ASSETS)
returns = get_returns(prices)

# Pesos do perfil medium
w_med = optimize_for_profile(prices, "medium")
print("Pesos medium:", {k: round(v, 3) for k, v in w_med.items()})

# VaR e CVaR
var_result = portfolio_var_cvar(w_med, returns, confidence=0.95, capital=10000)
print("\nVaR / CVaR (perfil MEDIUM, EUR 10.000):")
for k, v in var_result.items():
    print(f"  {k}: {v}")

# Stress tests
print("\nStress tests (perfil MEDIUM):")
stress = run_stress_tests(w_med, capital=10000)
print(stress[["cenario", "perda_%", "perda_eur", "valor_restante_eur"]].to_string(index=False))

# Comparar os 3 perfis em stress
print("\nComparação de stress tests:")
for profile in ["low", "medium", "high"]:
    w = optimize_for_profile(prices, profile)
    s = run_stress_tests(w, capital=10000)
    covid = s[s["cenario"].str.contains("COVID")].iloc[0]
    print(f"  {profile:6s}  COVID crash: {covid['perda_%']:+.1f}%  "
          f"(EUR {covid['perda_eur']:,.0f})")