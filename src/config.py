"""Project-wide constants: analysis window, seeds, paths, and the stock universe."""
from pathlib import Path

SEED = 42

# Fixed analysis window (May 2016 - April 2026) so results are reproducible.
START_DATE = "2016-05-01"
END_DATE = "2026-05-01"
TRADING_DAYS = 252

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

# 100 large-cap US stocks: 10 Information Technology names and 9 from each other GICS sector.
# Chosen as of 2026, so the universe carries survivorship bias (see README).
TICKERS_BY_SECTOR = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM", "TXN", "QCOM", "MU"],
    "Financials":             ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "CB", "PGR"],
    "Health Care":            ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "GILD"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "GM"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA"],
    "Industrials":            ["CAT", "HON", "UNP", "UPS", "BA", "RTX", "DE", "LMT", "GE"],
    "Consumer Staples":       ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "KMB"],
    "Energy":                 ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY"],
    "Utilities":              ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG"],
    "Real Estate":            ["PLD", "AMT", "EQIX", "CCI", "SPG", "O", "PSA", "WELL", "AVB"],
    "Materials":              ["LIN", "APD", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "ALB"],
}
