import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def fetch_euronext_stocks():
    """
    Fetch top Euronext stocks and verify their availability on Yahoo Finance
    Returns a dictionary of valid stock symbols grouped by market
    """
    # Initialize dictionary to store valid stocks
    valid_stocks = {
        'Euronext Paris': [],
        'Euronext Amsterdam': [],
        'Euronext Brussels': []
    }
    
    # Market extensions mapping
    market_extensions = {
        'Euronext Paris': '.PA',
        'Euronext Amsterdam': '.AS',
        'Euronext Brussels': '.BR'
    }
    
    def verify_yahoo_symbol(symbol):
        """Verify if a symbol is available on Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d")
            return not df.empty
        except Exception:
            return False

    # Test some major stocks from each market
    test_stocks = {
        'Euronext Paris': [
            'AI', 'BNP', 'MC', 'OR', 'SAN', 'VIV'
        ],
        'Euronext Amsterdam': [
            'ASML', 'RAND', 'UNA', 'AD', 'INGA', 'PHIA'
        ],
        'Euronext Brussels': [
            'SOLB', 'KBC', 'UCB', 'ABI', 'GLPG', 'PROX'
        ]
    }

    print("Testing stock availability on Yahoo Finance...")
    
    for market, stocks in test_stocks.items():
        print(f"\nTesting {market} stocks:")
        extension = market_extensions[market]
        
        for stock in stocks:
            symbol = f"{stock}{extension}"
            if verify_yahoo_symbol(symbol):
                valid_stocks[market].append(symbol)
                print(f"✓ {symbol} is valid")
            else:
                print(f"✗ {symbol} is not available")
            time.sleep(1)  # Avoid hitting rate limits
    
    return valid_stocks

if __name__ == "__main__":
    print("Fetching and verifying Euronext stocks...")
    valid_stocks = fetch_euronext_stocks()
    
    print("\nSummary of valid stocks:")
    for market, stocks in valid_stocks.items():
        print(f"\n{market}:")
        print(", ".join(stocks))
        
    print("\nYou can update the monitored_stocks dictionary in market_data.py with these verified symbols.")
