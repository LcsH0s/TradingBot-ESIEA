import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from wallet import Wallet
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import setup_logger, get_class_logger
import signal
import logging
import sys

@dataclass
class TradingSignal:
    ticker: str
    action: str  # 'buy' or 'sell'
    confidence: float  # 0 to 1
    strategy: str
    price: float
    timestamp: datetime

class TraderBot:
    def __init__(self, wallet: Wallet, tickers: List[str], 
                 min_confidence: float = 0.7,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bollinger_period: int = 20,
                 bollinger_std: float = 2.0,
                 logger: Optional[logging.Logger] = None):
        self.wallet = wallet
        self.tickers = tickers
        self.min_confidence = min_confidence
        
        # Technical indicators parameters
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        
        # Transaction cost
        self.transaction_cost = 0.0025  # 0.25%
        
        # Setup logger
        self.logger = get_class_logger(logger or setup_logger(), "TraderBot")
        self.logger.info(f"Initialized TraderBot with {len(tickers)} tickers")
        self.logger.debug(f"Parameters: RSI={rsi_period}, MACD={macd_fast}/{macd_slow}/{macd_signal}, BB={bollinger_period}/{bollinger_std}")
        
        # Running flag for graceful shutdown
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info("\nReceived interrupt signal. Starting graceful shutdown...")
        self.shutdown()

    def shutdown(self):
        """Perform graceful shutdown."""
        try:
            self.running = False
            self.logger.info("Stopping trading bot...")
            
            # Calculate final portfolio statistics
            total_value = self.wallet.get_portfolio_value()
            profit_loss = total_value - self.wallet.initial_balance
            profit_percentage = (profit_loss / self.wallet.initial_balance) * 100
            
            self.logger.info("\n=== Final Trading Statistics ===")
            self.logger.info(f"Initial Balance: ${self.wallet.initial_balance:,.2f}")
            self.logger.info(f"Final Portfolio Value: ${total_value:,.2f}")
            self.logger.info(f"Total Profit/Loss: ${profit_loss:,.2f} ({profit_percentage:,.2f}%)")
            
            # Shutdown wallet (cancels pending orders and saves state)
            self.wallet.shutdown()
            
            self.logger.info("Trading bot shutdown complete. Goodbye!")
            sys.exit(0)
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
            sys.exit(1)

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Relative Strength Index"""
        self.logger.debug("Calculating RSI")
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        self.logger.debug("Calculating MACD")
        exp1 = data['Close'].ewm(span=self.macd_fast).mean()
        exp2 = data['Close'].ewm(span=self.macd_slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal).mean()
        return macd, signal

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        self.logger.debug("Calculating Bollinger Bands")
        middle = data['Close'].rolling(window=self.bollinger_period).mean()
        std = data['Close'].rolling(window=self.bollinger_period).std()
        upper = middle + (std * self.bollinger_std)
        lower = middle - (std * self.bollinger_std)
        return upper, middle, lower

    def analyze_ticker(self, ticker: str) -> Optional[TradingSignal]:
        """Analyze a single ticker and generate trading signal"""
        self.logger.info(f"Analyzing ticker: {ticker}")
        try:
            # Get historical data
            data = self.wallet.get_stock_data(ticker)
            if len(data) < max(self.rsi_period, self.macd_slow, self.bollinger_period):
                self.logger.warning(f"Insufficient historical data for {ticker}")
                return None

            # Calculate technical indicators
            rsi = self.calculate_rsi(data)
            macd, signal = self.calculate_macd(data)
            upper, middle, lower = self.calculate_bollinger_bands(data)

            current_price = data['Close'].iloc[-1]
            self.logger.info(f"{ticker} current price: ${current_price:,.2f}")
            
            # Initialize confidence scores for each indicator
            scores = []
            strategies = []

            # RSI Analysis (more aggressive thresholds)
            current_rsi = rsi.iloc[-1]
            self.logger.info(f"{ticker} RSI: {current_rsi:.2f}")
            if current_rsi < 35:  # Changed from 30
                score = 0.6 + (35 - current_rsi) / 100  # Changed base score from 0.7
                scores.append(score)
                strategies.append("RSI_OVERSOLD")
                self.logger.info(f"RSI Oversold signal: {score:.2f}")
            elif current_rsi > 65:  # Changed from 70
                score = 0.6 + (current_rsi - 65) / 100  # Changed base score from 0.7
                scores.append(score)
                strategies.append("RSI_OVERBOUGHT")
                self.logger.info(f"RSI Overbought signal: {score:.2f}")

            # MACD Analysis (more sensitive)
            macd_current = macd.iloc[-1]
            signal_current = signal.iloc[-1]
            macd_prev = macd.iloc[-2]
            signal_prev = signal.iloc[-2]
            
            self.logger.info(f"{ticker} MACD: {macd_current:.4f}, Signal: {signal_current:.4f}")
            
            if macd_current > signal_current and macd_prev <= signal_prev:
                scores.append(0.7)  # Changed from 0.8
                strategies.append("MACD_BULLISH")
                self.logger.info("MACD Bullish crossover detected")
            elif macd_current < signal_current and macd_prev >= signal_prev:
                scores.append(0.7)  # Changed from 0.8
                strategies.append("MACD_BEARISH")
                self.logger.info("MACD Bearish crossover detected")

            # Bollinger Bands Analysis
            current_close = data['Close'].iloc[-1]
            lower_band = lower.iloc[-1]
            upper_band = upper.iloc[-1]
            
            self.logger.info(f"{ticker} BB: Lower={lower_band:.2f}, Current={current_close:.2f}, Upper={upper_band:.2f}")
            
            if current_close < lower_band:
                bb_score = 0.6 + (lower_band - current_close) / lower_band
                scores.append(bb_score)
                strategies.append("BB_OVERSOLD")
                self.logger.info(f"Price below lower Bollinger Band, score: {bb_score:.2f}")
            elif current_close > upper_band:
                bb_score = 0.6 + (current_close - upper_band) / upper_band
                scores.append(bb_score)
                strategies.append("BB_OVERBOUGHT")
                self.logger.info(f"Price above upper Bollinger Band, score: {bb_score:.2f}")

            if not scores:
                self.logger.info(f"No trading signals for {ticker}")
                return None

            # Calculate final confidence and determine action
            final_confidence = sum(scores) / len(scores)
            self.logger.info(f"Final confidence score: {final_confidence:.2f}")
            
            # Current position check
            current_position = self.wallet.get_position(ticker)
            
            # Determine action based on signals and current position
            if any(s.endswith("OVERSOLD") for s in strategies) and \
               (current_position is None or current_position.quantity <= 0):
                action = "buy"
            elif any(s.endswith("OVERBOUGHT") for s in strategies) and \
                 (current_position is not None and current_position.quantity > 0):
                action = "sell"
            else:
                self.logger.info(f"No actionable signals for {ticker}")
                return None

            # Consider transaction costs
            if action == "buy":
                required_increase = current_price * (1 + 2 * self.transaction_cost)
                if not any(p > required_increase for p in data['Close'].iloc[-10:]):
                    final_confidence *= 0.8
                    self.logger.info(f"Reduced confidence due to transaction costs: {final_confidence:.2f}")

            signal = TradingSignal(
                ticker=ticker,
                action=action,
                confidence=final_confidence,
                strategy="+".join(strategies),
                price=current_price,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Generated signal for {ticker}: {action.upper()} (confidence: {final_confidence:.2f})")
            return signal

        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
            return None

    def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal if confidence meets minimum threshold"""
        self.logger.info(f"Evaluating signal: {signal.ticker} {signal.action.upper()} (confidence: {signal.confidence:.2f}, min required: {self.min_confidence})")
        
        if signal.confidence < self.min_confidence:
            self.logger.info(f"Signal confidence ({signal.confidence:.2f}) below minimum threshold ({self.min_confidence})")
            return False

        try:
            position = self.wallet.get_position(signal.ticker)
            
            if signal.action == "buy":
                available_balance = self.wallet.balance * 0.2  # Changed from 0.1 to allow larger positions
                position_size = int(available_balance / signal.price)  # Removed confidence multiplier
                
                if position_size > 0:
                    self.logger.info(f"Executing BUY order: {position_size} shares of {signal.ticker} at ${signal.price:.2f}")
                    return self.wallet.place_order("buy", signal.ticker, position_size)
                else:
                    self.logger.info(f"Position size too small for {signal.ticker}")
                    
            elif signal.action == "sell" and position is not None:
                self.logger.info(f"Executing SELL order: {position.quantity} shares of {signal.ticker} at ${signal.price:.2f}")
                return self.wallet.place_order("sell", signal.ticker, position.quantity)
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.ticker}: {str(e)}", exc_info=True)
            return False

    def run(self, interval_seconds: int = 300):
        """Run the trading bot continuously"""
        self.logger.info(f"Starting trading bot with {len(self.tickers)} tickers...")
        self.logger.info(f"Check interval: {interval_seconds} seconds")
        self.logger.info("Press Ctrl+C to stop the bot gracefully")
        
        try:
            while self.running:
                for ticker in self.tickers:
                    if not self.running:
                        break
                        
                    try:
                        self.logger.debug(f"Processing ticker: {ticker}")
                        signal = self.analyze_ticker(ticker)
                        
                        if signal:
                            self.logger.info(f"Signal detected for {ticker}:")
                            self.logger.info(f"Action: {signal.action.upper()}")
                            self.logger.info(f"Confidence: {signal.confidence:.2f}")
                            self.logger.info(f"Strategy: {signal.strategy}")
                            self.logger.info(f"Price: ${signal.price:.2f}")
                            
                            if self.execute_signal(signal):
                                self.logger.info(f"Successfully executed {signal.action} order for {ticker}")
                            else:
                                self.logger.warning(f"Failed to execute {signal.action} order for {ticker}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
                        continue
                
                if self.running:
                    self.logger.debug(f"Sleeping for {interval_seconds} seconds")
                    time.sleep(interval_seconds)
                    
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
            self.shutdown()
