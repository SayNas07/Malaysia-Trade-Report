# Malaysian trade concentration and sensitivity to external shocks

This project measures how concentrated Malaysia's goods trade is, and how exports and imports respond to external shocks. Concentration is calculated from UN Comtrade. The shock analysis uses a quarterly VECM for the dynamics and single-equation ARDL models for the long-run demand elasticities.

## Data sources

- UN Comtrade annual HS trade, which needs a free API key
- World Bank World Development Indicators
- Caldara and Iacoviello trade-policy uncertainty
- OpenDOSM monthly goods trade and Malaysia's headline CPI
- BIS broad real effective exchange rate
- Partner real GDP: BEA (United States), Cabinet Office (Japan), Eurostat (EU27), SingStat (Singapore), and OECD plus NBS growth rates (China)
- FRED: the ringgit (EXMAUS), US CPI (CPIAUCSL), and the IMF all-commodity price index (PALLFNFINDEXQ)

`data/raw/` and `data/processed/` stay on your machine. They are not part of the git history.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set `COMTRADE_API_KEY` to the key from the [UN Comtrade developer portal](https://comtradedeveloper.un.org/). Leave the value empty only if you already have the Comtrade extracts in `data/raw/`.

## Run order

```bash
python src/fetch_data.py
python src/concentration.py
python src/var_data.py
python src/var_model.py
```

`fetch_data.py` downloads Comtrade, WDI, and trade-policy uncertainty. If a source fails, it prints the file to save under `data/raw/` and does not fill the gap. `concentration.py` reads those Comtrade extracts. `var_data.py` builds the quarterly file and selects the VECM lag and rank. `var_model.py` estimates the VECM, the ARDL demand equations, and writes `outputs/results_summary.md`.
