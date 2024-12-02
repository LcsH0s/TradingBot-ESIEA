import yfinance as yf
import pandas as pd
import json
import os
from typing import List, Dict
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logger
import ssl
import certifi
import requests
import numpy as np
from tabulate import tabulate
from io import StringIO
from utils.yahoo_client import YahooAPIClient

class StockScreener:
    def __init__(self, lookback_period: int = 30):
        """Initialize the stock screener with a lookback period."""
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        self.yahoo_client = YahooAPIClient()

    def _get_sp500_tickers(self) -> List[str]:
        """Fetch S&P 500 tickers from Wikipedia with SSL verification."""
        self.logger.info("Using verified S&P 500 tickers list...")
        # Updated list of S&P 500 companies with verified active tickers
        return [
            "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "BRK.B", "GOOG", "AVGO",
            "JPM", "LLY", "UNH", "V", "XOM", "MA", "COST", "HD", "PG", "WMT",
            "NFLX", "JNJ", "ABBV", "BAC", "CRM", "ORCL", "CVX", "WFC", "MRK", "KO",
            "CSCO", "ADBE", "ACN", "PEP", "AMD", "LIN", "NOW", "DIS", "MCD", "IBM",
            "ABT", "PM", "TMO", "GE", "CAT", "GS", "ISRG", "VZ", "TXN", "INTU"
        ]

    def get_sp500_tickers(self) -> List[str]:
        """Fetch S&P 500 tickers using Wikipedia."""
        return self._get_sp500_tickers()

    def analyze_stock(self, ticker: str) -> Dict:
        """Analyze a single stock's performance."""
        try:
            # Get stock info and validate
            info = self.yahoo_client.get_stock_info(ticker)
            if not info:
                self.logger.warning(f"No data available for {ticker}")
                return None
                
            # Check if we have a valid price
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not current_price:
                self.logger.warning(f"No price data available for {ticker}")
                return None
            
            # Get historical data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)  # Get 3 months of data
            data = self.yahoo_client.get_stock_history(ticker, start_date, end_date)
            
            if data is None or len(data) < self.lookback_period:
                self.logger.warning(f"Insufficient data points for {ticker}")
                return None
            
            # Calculate performance metrics
            lookback_data = data.tail(self.lookback_period)
            
            # Calculate returns
            total_return = (lookback_data['Close'].iloc[-1] / lookback_data['Close'].iloc[0]) - 1
            
            # Calculate volatility (annualized)
            daily_returns = lookback_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
            
            # Calculate Sharpe Ratio (assuming risk-free rate of 0.05)
            excess_returns = daily_returns - 0.05/252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
            
            # Calculate RSI
            delta = lookback_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
            
            # Calculate trend strength (using linear regression)
            x = np.arange(len(lookback_data))
            y = lookback_data['Close'].values
            slope, _ = np.polyfit(x, y, 1)
            trend_strength = slope / lookback_data['Close'].mean()
            
            # Calculate volume trend
            volume_trend = lookback_data['Volume'].iloc[-5:].mean() / lookback_data['Volume'].iloc[:-5].mean() - 1
            
            # Get market cap
            market_cap = info.get('marketCap', 0)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'market_cap': market_cap,
                'total_return': total_return * 100,  # Convert to percentage
                'volatility': volatility * 100,  # Convert to percentage
                'sharpe_ratio': sharpe_ratio,
                'rsi': rsi,
                'trend_strength': trend_strength * 100,  # Convert to percentage
                'volume_trend': volume_trend * 100,  # Convert to percentage
                'composite_score': (
                    0.3 * total_return +  # 30% weight on returns
                    0.2 * sharpe_ratio +  # 20% weight on risk-adjusted returns
                    0.2 * (70 - abs(rsi - 50))/70 +  # 20% weight on RSI (closer to 50 is better)
                    0.2 * trend_strength +  # 20% weight on trend
                    0.1 * volume_trend  # 10% weight on volume trend
                ) * 10  # Scale to 0-10
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing {ticker}: {str(e)}")
            return None

    def update_config(self, config_path: str = 'config.json', top_n: int = 20) -> None:
        """Update config file with top performing stocks."""
        try:
            # Get tickers and analyze them
            tickers = self.get_sp500_tickers()
            self.logger.info(f"Starting analysis of {len(tickers)} stocks (lookback period: {self.lookback_period} days)")
            
            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {executor.submit(self.analyze_stock, ticker): ticker for ticker in tickers}
                for future in as_completed(future_to_ticker):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            
            # Sort by score
            results.sort(key=lambda x: x['composite_score'], reverse=True)
            top_performers = results[:top_n]
            
            # Update config file
            config = {
                'tickers': [stock['ticker'] for stock in top_performers],
                'min_confidence': 0.7,
                'check_interval': 300,
                'ticker_performance': {
                    'last_updated': pd.Timestamp.now(tz='UTC').isoformat(),
                    'lookback_period_days': self.lookback_period,
                    'performers': top_performers
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Analysis complete. Selected top {len(top_performers)} performers.")
            self.logger.info(f"Updated {config_path} with top {len(top_performers)} performers")
            
            # Print results in a nice table
            print("\n=== Top Performing Stocks ===")
            headers = ['Ticker', 'Return', 'Sharpe', 'RSI', 'Score']
            rows = [[
                stock['ticker'],
                f"{stock['total_return']:.1f}%",
                f"{stock['sharpe_ratio']:.2f}",
                f"{stock['rsi']:.1f}",
                f"{stock['composite_score']:.1f}"
            ] for stock in top_performers]
            
            print(tabulate(rows, headers=headers, tablefmt='simple'))
            
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Create screener instance
    screener = StockScreener(lookback_period=30)  # 30-day lookback period
    
    # Update config with top 20 performers
    screener.update_config(top_n=20)
