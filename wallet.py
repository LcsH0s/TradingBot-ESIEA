import json
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Literal, Optional, Tuple, Any
from dataclasses import dataclass
import os
import logging
from utils.logger import setup_logger, get_class_logger
from utils.yahoo_client import YahooAPIClient
import yfinance as yf
import pickle
from pathlib import Path

@dataclass
class Trade:
    type: Literal["buy", "sell"]
    ticker: str
    quantity: float
    price: float
    date: str
    status: Literal["pending", "executed", "failed", "cancelled"] = "pending"
    execution_date: Optional[str] = None

class Position:
    def __init__(self, ticker: str, quantity: float, avg_price: float):
        self.ticker = ticker
        self.quantity = quantity
        self.avg_price = avg_price

class StockCache:
    def __init__(self, cache_duration: int = 120, cache_file: str = "stock_cache.pkl", logger: Optional[logging.Logger] = None):
        self.cache_duration = cache_duration
        self.cache_file = cache_file
        self.cache_lock = threading.Lock()
        self.logger = get_class_logger(logger or setup_logger(), "StockCache")
        self.running = True
        
        # Initialize cache dictionaries
        self.cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self.price_cache: Dict[str, Tuple[float, float]] = {}
        self.news_cache: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
        
        # Load cache from file if it exists
        self._load_cache()
        
        # Start cache cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_cache, daemon=True)
        self.cleanup_thread.start()
        
    def _load_cache(self):
        """Load cache from file if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.cache = cached_data.get('historical', {})
                    self.price_cache = cached_data.get('price', {})
                    self.news_cache = cached_data.get('news', {})
                    
                # Immediately cleanup any stale data
                self._cleanup_cache_data()
        except Exception as e:
            self.logger.error(f"Error loading cache from file: {str(e)}")
            # Start fresh if cache file is corrupted
            self._save_cache()
            
    def _save_cache(self):
        """Save cache to file."""
        try:
            # Try to acquire the lock with a timeout of 5 seconds
            if not self.cache_lock.acquire(timeout=5):
                self.logger.warning("Could not acquire cache lock for saving - skipping this save")
                return
            try:
                cache_data = {
                    'historical': self.cache,
                    'price': self.price_cache,
                    'news': self.news_cache
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            finally:
                self.cache_lock.release()
        except Exception as e:
            self.logger.error(f"Error saving cache to file: {str(e)}")
            
    def _cleanup_cache_data(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        needs_save = False
        
        try:
            if not self.cache_lock.acquire(timeout=5):
                self.logger.warning("Could not acquire cache lock for cleanup - skipping this cleanup")
                return
            try:
                # Cleanup historical data cache
                expired_historical = [
                    ticker for ticker, (_, timestamp) in self.cache.items()
                    if current_time - timestamp > self.cache_duration
                ]
                for ticker in expired_historical:
                    del self.cache[ticker]
                    
                # Cleanup price cache
                expired_prices = [
                    ticker for ticker, (_, timestamp) in self.price_cache.items()
                    if current_time - timestamp > self.cache_duration
                ]
                for ticker in expired_prices:
                    del self.price_cache[ticker]
                    
                # Cleanup news cache
                expired_news = [
                    ticker for ticker, (_, timestamp) in self.news_cache.items()
                    if current_time - timestamp > self.cache_duration
                ]
                for ticker in expired_news:
                    del self.news_cache[ticker]
                
                needs_save = bool(expired_historical or expired_prices or expired_news)
            finally:
                self.cache_lock.release()
                
            # Save cleaned cache to file outside of the lock
            if needs_save:
                self._save_cache()
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {str(e)}")
            
    def _cleanup_old_cache(self):
        """Background thread to periodically clean up old cache entries."""
        while self.running:
            try:
                self._cleanup_cache_data()
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {str(e)}")
            
            # Sleep in small intervals to check running flag more frequently
            for _ in range(30):  # 30 iterations of 1-second sleep
                if not self.running:
                    break
                time.sleep(1)
        
        self.logger.info("Cache cleanup thread stopped")

    def shutdown(self):
        """Stop the cache cleanup thread."""
        self.logger.debug("Shutting down cache cleanup thread...")
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)  # Wait up to 5 seconds for thread to stop
        self.logger.debug("Cache cleanup thread stopped")

    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        with self.cache_lock:
            if ticker in self.cache:
                data, timestamp = self.cache[ticker]
                if time.time() - timestamp < self.cache_duration:
                    return data
        return None

    def set_data(self, ticker: str, data: pd.DataFrame):
        try:
            self.logger.debug(f"Acquiring cache lock for setting data for {ticker}")
            with self.cache_lock:
                self.logger.debug(f"Caching data for {ticker}: {len(data)} rows")
                self.cache[ticker] = (data, time.time())
            self.logger.debug("Released cache lock")
            # Save cache after releasing the lock
            self._save_cache()
            self.logger.debug(f"Successfully cached data for {ticker}")
        except Exception as e:
            self.logger.error(f"Error setting cache data: {str(e)}")

    def get_price(self, ticker: str) -> Optional[float]:
        with self.cache_lock:
            if ticker in self.price_cache:
                price, timestamp = self.price_cache[ticker]
                if time.time() - timestamp < self.cache_duration:
                    return price
        return None

    def set_price(self, ticker: str, price: float):
        self.logger.debug(f"Acquiring cache lock for setting price for {ticker}")
        with self.cache_lock:
            self.price_cache[ticker] = (price, time.time())
        self.logger.debug("Released cache lock")
        self._save_cache()

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        with self.cache_lock:
            if ticker in self.news_cache:
                news, timestamp = self.news_cache[ticker]
                if time.time() - timestamp < self.cache_duration:
                    return news
        return None

    def set_news(self, ticker: str, news: List[Dict[str, Any]]):
        self.logger.debug(f"Acquiring cache lock for setting news for {ticker}")
        with self.cache_lock:
            self.news_cache[ticker] = (news, time.time())
        self.logger.debug("Released cache lock")
        self._save_cache()

class Wallet:
    def __init__(self, initial_balance: float = 10000.0, wallet_file: str = "wallet.json", logger: Optional[logging.Logger] = None):
        """Initialize wallet with optional initial balance."""
        self.wallet_file = wallet_file
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.trade_lock = threading.Lock()
        self.FEE_RATE = 0.0025
        self.cache = StockCache(cache_duration=60, logger=logger)  # Set cache duration to 60 seconds
        self.logger = get_class_logger(logger or setup_logger(), "Wallet")
        self.running = True
        self.yahoo_client = YahooAPIClient(logger=logger)
        
        # Load existing wallet if it exists, otherwise use initial values
        if os.path.exists(wallet_file):
            self._load_wallet()
        else:
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self._save_wallet()
        
        # Start the order execution thread
        self.execution_thread = threading.Thread(target=self._execute_pending_trades, daemon=True)
        self.execution_thread.start()
        self.logger.info("Wallet initialized and execution thread started")

    def _load_wallet(self) -> None:
        """Load wallet data from JSON file if it exists."""
        try:
            with open(self.wallet_file, 'r') as f:
                data = json.load(f)
                self.initial_balance = data.get('initial_balance', 10000.0)
                self.balance = data.get('balance', self.initial_balance)
                
                # Load positions
                self.positions = {}
                for ticker, pos_data in data.get('positions', {}).items():
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=pos_data['quantity'],
                        avg_price=pos_data['avg_price']
                    )
                
                # Load trades
                self.trades = []
                for trade_data in data.get('trades', []):
                    self.trades.append(Trade(**trade_data))
                    
        except Exception as e:
            self.logger.error(f"Error loading wallet data: {str(e)}")
            # Initialize with default values if loading fails
            self.initial_balance = 10000.0
            self.balance = self.initial_balance
            self.positions = {}
            self.trades = []

    def _save_wallet(self) -> None:
        """Save wallet data to JSON file."""
        try:
            self.logger.debug("Acquiring trade lock for saving wallet")
            with self.trade_lock:
                data = {
                    'initial_balance': self.initial_balance,
                    'balance': self.balance,
                    'positions': {
                        ticker: {'quantity': pos.quantity, 'avg_price': pos.avg_price}
                        for ticker, pos in self.positions.items()
                    },
                    'trades': [vars(trade) for trade in self.trades]
                }
                
                # Write to a temporary file first
                temp_file = f"{self.wallet_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=4)
                
                # Then atomically rename it to the actual file
                os.replace(temp_file, self.wallet_file)
            self.logger.debug("Released trade lock")
        except Exception as e:
            self.logger.error(f"Error saving wallet data: {str(e)}")
            # If there was an error, try to restore from the original file
            try:
                if os.path.exists(f"{self.wallet_file}.tmp"):
                    os.remove(f"{self.wallet_file}.tmp")
            except:
                pass

    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance with caching."""
        self.logger.debug(f"Fetching stock data for {ticker}")
        
        # Check cache first
        cached_data = self.cache.get_data(ticker)
        if cached_data is not None:
            self.logger.debug(f"Using cached data for {ticker} (age: {time.time() - self.cache.cache[ticker][1]:.1f}s)")
            return cached_data
            
        try:
            # Get historical data for the last 60 days to ensure enough data for technical analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Use last 60 days instead of 30
            hist = self.yahoo_client.get_stock_history(ticker, start_date, end_date)
            
            if hist is None or hist.empty:
                raise ValueError(f"No historical data available for {ticker}")
                
            # Cache the data
            self.logger.debug(f"Got historical data for {ticker}: {len(hist)} rows, columns: {hist.columns.tolist()}")
            self.cache.set_data(ticker, hist)
            self.logger.debug(f"Successfully fetched and cached {len(hist)} data points for {ticker}")
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    def get_current_price(self, ticker: str, force_update: bool = False) -> float:
        """Get the current price of a stock with caching."""
        self.logger.debug(f"Fetching current price for {ticker}")
        
        # Check cache first
        cached_price = self.cache.get_price(ticker)
        if cached_price is not None and not force_update:
            self.logger.debug(f"Using cached price for {ticker} (age: {time.time() - self.cache.price_cache[ticker][1]:.1f}s): ${cached_price:,.2f}")
            return cached_price
            
        try:
            # Get stock info from Yahoo client
            info = self.yahoo_client.get_stock_info(ticker)
            if not info:
                raise ValueError(f"No price data available for {ticker}")
                
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not price:
                raise ValueError(f"Invalid price data for {ticker}")
                
            # Cache the price
            self.cache.set_price(ticker, price)
            self.logger.debug(f"Current price for {ticker}: ${price:,.2f}")
            return price
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {ticker}: {str(e)}")
            raise

    def get_stock_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Get recent news for a stock with caching."""
        self.logger.debug(f"Fetching news for {ticker}")
        
        # Check cache first
        cached_news = self.cache.get_news(ticker)
        if cached_news is not None:
            self.logger.debug(f"Using cached news for {ticker}")
            return cached_news
            
        try:
            # Get stock info which includes news
            info = self.yahoo_client.get_stock_info(ticker)
            if not info:
                raise ValueError(f"No data available for {ticker}")
                
            news = info.get('news', [])
            
            # Cache the news
            self.cache.set_news(ticker, news)
            self.logger.debug(f"Successfully fetched {len(news)} news items for {ticker}")
            return news
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []

    def place_order(self, order_type: Literal["buy", "sell"], ticker: str, quantity: float) -> bool:
        """Place a new order."""
        self.logger.info(f"Placing {order_type} order for {quantity} shares of {ticker}")
        current_price = self.get_current_price(ticker)
        total_cost = current_price * quantity * (1 + self.FEE_RATE)
        
        self.logger.debug("Acquiring trade lock for placing order")
        with self.trade_lock:
            if order_type == "buy":
                if total_cost > self.balance:
                    self.logger.warning(f"Insufficient funds for buy order. Required: ${total_cost:,.2f}, Available: ${self.balance:,.2f}")
                    return False
            
            if order_type == "sell":
                position = self.positions.get(ticker)
                if not position or position.quantity < quantity:
                    self.logger.warning(f"Insufficient shares for sell order. Required: {quantity}, Available: {position.quantity if position else 0}")
                    return False

            # Create and append the trade
            trade = Trade(
                type=order_type,
                ticker=ticker,
                quantity=quantity,
                price=current_price,
                date=datetime.now().isoformat()
            )
            self.trades.append(trade)
            
            # Save immediately after modifying trades list
            self.logger.info(f"Order placed successfully: {order_type} {quantity} {ticker} @ ${current_price:,.2f}")
            return True
        self.logger.debug("Released trade lock")
        self._save_wallet()
        self.logger.debug("Wallet saved")

    def cancel_pending_orders(self) -> None:
        """Cancel all pending orders and refund them."""

        self.logger.info("Cancelling all pending orders")
        self.logger.debug("Acquiring trade lock for cancelling orders")
        with self.trade_lock:
            modified = False
            for trade in self.trades:
                if trade.status == "pending":
                    # if trade.type == "buy":
                    #     # Refund the reserved amount including fees
                    #     refund_amount = trade.price * trade.quantity * (1 + self.FEE_RATE)
                    #     self.balance += refund_amount
                    #     self.logger.info(f"Refunded ${refund_amount:,.2f} for cancelled {trade.type} order of {trade.quantity} {trade.ticker}")
                    
                    trade.status = "cancelled"
                    trade.execution_date = datetime.now().isoformat()
                    modified = True
        self.logger.debug("Released trade lock")
        if modified:
            self._save_wallet()
            self.logger.info("All pending orders cancelled and wallet saved")
        else:
            self.logger.info("No pending orders to cancel")

    def shutdown(self) -> None:
        """Gracefully shutdown the wallet."""
        self.logger.info("Initiating wallet shutdown")
        
        # First set running to False to stop background operations
        self.running = False
        self.cache.running = False
        
        try:
            # Wait for execution thread to finish with a timeout
            if hasattr(self, 'execution_thread') and self.execution_thread.is_alive():
                self.execution_thread.join(timeout=2)
            
            self.cancel_pending_orders()
        
            # Save final wallet state
            self._save_wallet()
            self.logger.info("Wallet shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during wallet shutdown: {str(e)}")
            # Continue with shutdown even if there was an error

    def _execute_pending_trades(self):
        """Background thread to execute pending trades after 1-minute delay."""
        self.logger.info("Starting trade execution thread")
        while self.running:
            try:
                changes_made = False
                with self.trade_lock:
                    current_time = datetime.now()
                    for trade in self.trades:
                        if not self.running:
                            break
                        if trade.status == "pending":
                            trade_time = datetime.fromisoformat(trade.date)
                            if (current_time - trade_time).total_seconds() >= 60:
                                self.logger.debug(f"Executing pending trade: {trade.type} {trade.quantity} {trade.ticker}")
                                self._execute_trade(trade)
                                changes_made = True
                if changes_made:
                    self._save_wallet()
            except Exception as e:
                self.logger.error(f"Error in trade execution thread: {str(e)}")
                if not self.running:  # If we're shutting down, exit the thread
                    break
            
            time.sleep(2)
        
        self.logger.info("Trade execution thread stopped")

    def _execute_trade(self, trade: Trade):
        """Execute a trade after the delay."""
        self.logger.info(f"Executing trade: {trade.type} {trade.quantity} {trade.ticker}")
        
        try:
            current_price = self.get_current_price(trade.ticker)
            
            total_cost = current_price * trade.quantity * (1 + self.FEE_RATE)
            
            if trade.type == "buy":
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    if trade.ticker in self.positions:
                        pos = self.positions[trade.ticker]
                        total_quantity = pos.quantity + trade.quantity
                        total_cost = (pos.quantity * pos.avg_price) + (trade.quantity * current_price)
                        pos.avg_price = total_cost / total_quantity
                        pos.quantity = total_quantity
                    else:
                        self.positions[trade.ticker] = Position(trade.ticker, trade.quantity, current_price)
                    trade.status = "executed"
                    self.logger.info(f"Buy order executed: {trade.quantity} {trade.ticker} @ ${current_price:,.2f}")
                else:
                    trade.status = "failed"
                    self.logger.error(f"Buy order failed: Insufficient funds")
            
            elif trade.type == "sell":
                if trade.ticker in self.positions and self.positions[trade.ticker].quantity >= trade.quantity:
                    self.balance += current_price * trade.quantity * (1 - self.FEE_RATE)
                    pos = self.positions[trade.ticker]
                    pos.quantity -= trade.quantity
                    if pos.quantity == 0:
                        del self.positions[trade.ticker]
                    trade.status = "executed"
                    self.logger.info(f"Sell order executed: {trade.quantity} {trade.ticker} @ ${current_price:,.2f}")
                else:
                    trade.status = "failed"
                    self.logger.error(f"Sell order failed: Insufficient shares")
            
            trade.execution_date = datetime.now().isoformat()
        
        except Exception as e:
            self.logger.error(f"Trade execution failed with error: {str(e)}", exc_info=True)
            self._save_wallet()

    def get_portfolio_value(self) -> float:
        """Get total portfolio value including cash balance."""
        total_value = self.balance
        for ticker, position in self.positions.items():
            current_price = self.get_current_price(ticker)
            total_value += current_price * position.quantity
        self.logger.debug(f"Portfolio value: ${total_value:,.2f}")
        return total_value

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get current position for a specific ticker."""
        position = self.positions.get(ticker)
        if position:
            self.logger.debug(f"Position for {ticker}: Quantity: {position.quantity}, Avg Price: ${position.avg_price:,.2f}")
        return position

    def get_trade_history(self) -> List[Trade]:
        """Get all historical trades."""
        self.logger.debug(f"Retrieving {len(self.trades)} historical trades")
        return self.trades.copy()
