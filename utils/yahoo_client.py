import yfinance as yf
import requests
import pandas as pd
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.logger import setup_logger, get_class_logger

class YahooAPIClient:
    """Client for interacting with Yahoo Finance API with fallback mechanisms."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = get_class_logger(logger or setup_logger(), "YahooAPI")
        self.session = self._create_session()
        self.base_url = "https://query2.finance.yahoo.com"
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic and proper headers."""
        session = requests.Session()
        
        # Configure retry strategy with shorter timeouts
        retry_strategy = Retry(
            total=2,  # Reduce total retries
            backoff_factor=0.5,  # Reduce backoff time
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers to mimic a browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        return session
        
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get stock information with fallback mechanisms."""
        try:
            self.logger.debug(f"Attempting to fetch stock info for {ticker} using yfinance")
            # Try yfinance first with a timeout
            stock = yf.Ticker(ticker)
            try:
                info = stock.info
                if info and len(info) > 0:
                    self.logger.debug(f"Successfully fetched {ticker} info from yfinance with {len(info)} fields")
                    return info
            except Exception as e:
                self.logger.warning(f"yfinance failed for {ticker}, trying direct API: {str(e)}")
        
            # Fallback to direct API request with timeout
            self.logger.debug(f"Attempting direct API request for {ticker} quote data")
            url = f"{self.base_url}/v8/finance/quote"
            params = {'symbols': ticker}
            
            response = self.session.get(url, params=params, timeout=5)  # 5 second timeout
            response.raise_for_status()
            
            data = response.json()
            if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                results = data['quoteResponse']['result']
                if results and len(results) > 0:
                    self.logger.debug(f"Successfully fetched {ticker} quote data from direct API")
                    return results[0]
        
            self.logger.warning(f"No data found for {ticker}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error fetching info for {ticker}: {str(e)}")
            return None
            
    def get_stock_history(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical stock data with fallback mechanisms."""
        try:
            self.logger.debug(f"Attempting to fetch history for {ticker} from {start_date} to {end_date} using yfinance")
            # Try yfinance first with a timeout
            stock = yf.Ticker(ticker)
            try:
                df = stock.history(start=start_date, end=end_date)
                if not df.empty:
                    self.logger.debug(f"Successfully fetched {ticker} history from yfinance: {len(df)} rows")
                    self.logger.debug(f"Data columns: {df.columns.tolist()}")
                    self.logger.debug(f"First row: {df.iloc[0].to_dict()}")
                    return df
            except Exception as e:
                self.logger.warning(f"yfinance history failed for {ticker}, trying direct API: {str(e)}")
        
            # Fallback to direct API request
            self.logger.debug(f"Attempting direct API request for {ticker} historical data")
            interval = '1d'
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            url = f"{self.base_url}/v8/finance/chart/{ticker}"
            params = {
                'period1': start_timestamp,
                'period2': end_timestamp,
                'interval': interval,
                'includeAdjustedClose': True
            }
            
            self.logger.debug(f"Making request to {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=5)  # 5 second timeout
            response.raise_for_status()
            
            data = response.json()
            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                
                # Extract time series data
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'Open': quotes['open'],
                    'High': quotes['high'],
                    'Low': quotes['low'],
                    'Close': quotes['close'],
                    'Volume': quotes['volume']
                }, index=pd.to_datetime(timestamps, unit='s'))
                
                if 'adjclose' in result['indicators']:
                    df['Adj Close'] = result['indicators']['adjclose'][0]['adjclose']
                    self.logger.debug(f"Added adjusted close prices to {ticker} data")
                
                self.logger.debug(f"Successfully fetched {ticker} history from direct API: {len(df)} rows")
                return df
                
            self.logger.warning(f"No historical data found for {ticker}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error fetching history for {ticker}: {str(e)}")
            return None

    def __del__(self):
        """Cleanup the session on deletion."""
        if self.session:
            self.session.close()
