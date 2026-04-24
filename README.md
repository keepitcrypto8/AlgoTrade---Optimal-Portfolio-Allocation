# AlgoTrade---Optimal-Portfolio-Allocation

Ferramenta de backtesting e gestão activa de portfolios multi-asset.

## Objectivo
Construir uma plataforma que permite:
- Definir qualquer conjunto de ativos (acções, ETFs, cripto, commodities)
- Enquadrá-los em categorias e definir rácios de investimento por categoria
- Testar diferentes estratégias de alocação e tipos de gestão (backtest)
- Comparar 3 perfis de risco: Low / Medium / High
- Ver qual a distribuição que maximiza o retorno ajustado ao risco

## Stack
Python 3.10+ · pandas · numpy · scipy · plotly · streamlit · yfinance

## Estrutura
src/          # módulos Python (dados, optimização, backtest, risco)
notebooks/    # análise exploratória e testes
app/          # dashboard Streamlit
data/         # preços em cache (não commitados)
results/      # outputs de backtest

## Como correr
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
