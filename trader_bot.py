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
        """Gracefully shutdown the trading bot."""
        self.logger.info("Shutting down trading bot...")
        self.running = False
        self.wallet.shutdown()  # Stop the wallet and its components
        self.logger.info("Trading bot shutdown complete")

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            self.logger.debug("Calculating RSI")
            if data.empty:
                raise ValueError("Empty dataframe provided")
            
            self.logger.debug(f"Data shape before RSI: {data.shape}")
            self.logger.debug(f"Data columns: {data.columns.tolist()}")
            self.logger.debug(f"First few close prices: {data['Close'].head().tolist()}")
            
            delta = data['Close'].diff()
            self.logger.debug(f"Delta calculated, first few values: {delta.head().tolist()}")
            
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            self.logger.debug(f"Gain calculated, first few values: {gain.head().tolist()}")
            
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            self.logger.debug(f"Loss calculated, first few values: {loss.head().tolist()}")
            
            # Handle division by zero
            rs = gain / loss.replace(0, float('inf'))
            self.logger.debug(f"RS calculated, first few values: {rs.head().tolist()}")
            
            rsi = 100 - (100 / (1 + rs))
            self.logger.debug(f"RSI calculated, first few values: {rsi.head().tolist()}")
            
            if rsi.isna().all():
                raise ValueError("RSI calculation resulted in all NaN values")
                
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            self.logger.error(f"Data info: {data.info()}")
            raise

    def calculate_macd(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        try:
            self.logger.debug("Calculating MACD")
            if data.empty:
                raise ValueError("Empty dataframe provided")
            
            exp1 = data['Close'].ewm(span=self.macd_fast).mean()
            exp2 = data['Close'].ewm(span=self.macd_slow).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.macd_signal).mean()
            
            if macd.isna().all() or signal.isna().all():
                raise ValueError("MACD calculation resulted in all NaN values")
                
            return macd, signal
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            self.logger.debug("Calculating Bollinger Bands")
            if data.empty:
                raise ValueError("Empty dataframe provided")
            
            middle = data['Close'].rolling(window=self.bollinger_period).mean()
            std = data['Close'].rolling(window=self.bollinger_period).std()
            upper = middle + (std * self.bollinger_std)
            lower = middle - (std * self.bollinger_std)
            
            if middle.isna().all() or upper.isna().all() or lower.isna().all():
                raise ValueError("Bollinger Bands calculation resulted in all NaN values")
                
            return upper, middle, lower
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def analyze_ticker(self, ticker: str) -> Optional[TradingSignal]:
        """Analyze a single ticker and generate trading signal"""
        try:
            self.logger.info(f"=== Starting analysis for {ticker} ===")
            
            # Get historical data
            self.logger.debug(f"[Step 1/7] Getting historical data for {ticker}")
            data = self.wallet.get_stock_data(ticker)
            self.logger.debug(f"Got historical data for {ticker}: Shape={data.shape}, Columns={data.columns.tolist()}")
            
            if data.empty:
                self.logger.warning(f"Empty data received for {ticker}")
                return None
                
            if len(data) < max(self.rsi_period, self.macd_slow, self.bollinger_period):
                self.logger.warning(f"Insufficient historical data for {ticker}: {len(data)} rows")
                return None

            # Calculate technical indicators
            self.logger.debug(f"[Step 2/7] Starting technical analysis for {ticker}")
            self.logger.debug(f"Last 5 closing prices: {data['Close'].tail().tolist()}")
            
            try:
                self.logger.debug(f"[Step 3/7] Calculating RSI for {ticker}")
                rsi = self.calculate_rsi(data)
                self.logger.debug(f"RSI calculation complete. Last 5 values: {rsi.tail().tolist()}")
            except Exception as e:
                self.logger.error(f"RSI calculation failed for {ticker}: {str(e)}")
                raise

            try:
                self.logger.debug(f"[Step 4/7] Calculating MACD for {ticker}")
                macd, signal = self.calculate_macd(data)
                self.logger.debug(f"MACD calculation complete. Last 5 values - MACD: {macd.tail().tolist()}, Signal: {signal.tail().tolist()}")
            except Exception as e:
                self.logger.error(f"MACD calculation failed for {ticker}: {str(e)}")
                raise

            try:
                self.logger.debug(f"[Step 5/7] Calculating Bollinger Bands for {ticker}")
                upper, middle, lower = self.calculate_bollinger_bands(data)
                self.logger.debug(f"Bollinger Bands calculation complete. Last values - Upper: {upper.iloc[-1]:.2f}, Middle: {middle.iloc[-1]:.2f}, Lower: {lower.iloc[-1]:.2f}")
            except Exception as e:
                self.logger.error(f"Bollinger Bands calculation failed for {ticker}: {str(e)}")
                raise

            self.logger.debug(f"[Step 6/7] Analyzing signals for {ticker}")
            current_price = data['Close'].iloc[-1]
            self.logger.debug(f"{ticker} current price: ${current_price:,.2f}")
            
            # Initialize confidence scores for each indicator
            scores = []
            strategies = []

            # RSI Analysis
            current_rsi = rsi.iloc[-1]
            self.logger.debug(f"{ticker} RSI: {current_rsi:.2f}")
            if current_rsi < 35:
                score = 0.6 + (35 - current_rsi) / 100
                scores.append(score)
                strategies.append("RSI_OVERSOLD")
                self.logger.debug(f"RSI Oversold signal: {score:.2f}")
            elif current_rsi > 65:
                score = 0.6 + (current_rsi - 65) / 100
                scores.append(score)
                strategies.append("RSI_OVERBOUGHT")
                self.logger.debug(f"RSI Overbought signal: {score:.2f}")

            # MACD Analysis
            macd_current = macd.iloc[-1]
            signal_current = signal.iloc[-1]
            macd_prev = macd.iloc[-2]
            signal_prev = signal.iloc[-2]
            
            self.logger.debug(f"{ticker} MACD: {macd_current:.4f}, Signal: {signal_current:.4f}")
            
            if macd_current > signal_current and macd_prev <= signal_prev:
                scores.append(0.7)
                strategies.append("MACD_BULLISH")
                self.logger.debug("MACD Bullish crossover detected")
            elif macd_current < signal_current and macd_prev >= signal_prev:
                scores.append(0.7)
                strategies.append("MACD_BEARISH")
                self.logger.debug("MACD Bearish crossover detected")

            # Bollinger Bands Analysis
            current_close = data['Close'].iloc[-1]
            lower_band = lower.iloc[-1]
            upper_band = upper.iloc[-1]
            
            self.logger.debug(f"{ticker} BB: Lower={lower_band:.2f}, Current={current_close:.2f}, Upper={upper_band:.2f}")
            
            if current_close < lower_band:
                score = 0.6 + (lower_band - current_close) / current_close
                scores.append(min(score, 0.9))  # Cap at 0.9
                strategies.append("BB_OVERSOLD")
                self.logger.debug(f"BB Oversold signal: {score:.2f}")
            elif current_close > upper_band:
                score = 0.6 + (current_close - upper_band) / current_close
                scores.append(min(score, 0.9))  # Cap at 0.9
                strategies.append("BB_OVERBOUGHT")
                self.logger.debug(f"BB Overbought signal: {score:.2f}")

            self.logger.debug(f"[Step 7/7] Generating final signal for {ticker}")
            # Generate trading signal if we have any scores
            if not scores:
                self.logger.info(f"No trading signals detected for {ticker}")
                return None

            # Determine action based on strategies
            if any(s.endswith("OVERSOLD") or s.endswith("BULLISH") for s in strategies):
                action = "buy"
            else:
                action = "sell"

            # Calculate final confidence as average of all scores
            final_confidence = sum(scores) / len(scores)
            
            signal = TradingSignal(
                ticker=ticker,
                action=action,
                confidence=final_confidence,
                strategy=", ".join(strategies),
                price=current_price,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"=== Analysis complete for {ticker} ===")
            self.logger.info(f"Generated signal: {action.upper()} (confidence: {final_confidence:.2f})")
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
