import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from wallet import Wallet
import nltk
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
import json
import os

# Set NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

class Trader:
    def __init__(self, wallet, rsi_period=14, short_window=12, long_window=26):
        """
        Initialize trader with technical analysis parameters
        :param wallet: Wallet instance to manage trades
        :param rsi_period: Period for RSI calculation
        :param short_window: Short-term period for MACD
        :param long_window: Long-term period for MACD
        """
        self.wallet = wallet
        self.rsi_period = rsi_period
        self.short_window = short_window
        self.long_window = long_window
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        logging.basicConfig(
            filename=f'logs/trader_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def clean_symbol(self, symbol):
        """Clean symbol by removing $ prefix and any whitespace"""
        if not symbol:
            return None
        return symbol.replace('$', '').replace(' ', '')

    def get_historical_data(self, symbol, period="1d", interval="1m"):
        """Fetch historical data for analysis"""
        try:
            # Clean symbol
            symbol = self.clean_symbol(symbol)
            if not symbol:
                logging.error("Invalid symbol provided")
                return None
                
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period).mean()  # Using EMA for faster calculation
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = data['Close'].ewm(span=self.short_window, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.long_window, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(self, data, window=10):  # Reduced window for faster signals
        """Calculate Bollinger Bands"""
        sma = data['Close'].ewm(span=window).mean()  # Using EMA for faster calculation
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    def get_news_sentiment(self, symbol, max_news=5):  # Reduced news items for faster analysis
        """
        Analyze sentiment from recent news articles
        Returns: dict with sentiment scores and recent headlines
        """
        try:
            # Clean symbol
            symbol = self.clean_symbol(symbol)
            if not symbol:
                logging.error("Invalid symbol provided")
                return None
                
            # Get news from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news = ticker.news[:max_news]
            
            if not news:
                return {
                    'sentiment_score': 0,
                    'headlines': [],
                    'sentiment': 'neutral'
                }

            # Analyze sentiment for each news item
            sentiments = []
            headlines = []
            
            for article in news:
                # Get the title and summary
                title = article.get('title', '')
                
                # Only analyze title for speed
                blob = TextBlob(title)
                
                # Get sentiment polarity (-1 to 1)
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment)
                
                headlines.append({
                    'title': title,
                    'sentiment': sentiment,
                    'date': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M:%S')
                })

            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Determine overall sentiment
            if avg_sentiment > 0.2:
                sentiment = 'positive'
            elif avg_sentiment < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment_score': avg_sentiment,
                'headlines': headlines,
                'sentiment': sentiment
            }

        except Exception as e:
            logging.error(f"Error analyzing news sentiment for {symbol}: {str(e)}")
            return None

    def analyze_stock(self, symbol):
        """
        Analyze stock and generate trading signals
        Returns: dict with analysis results and trading recommendation
        """
        # Clean symbol
        symbol = self.clean_symbol(symbol)
        if not symbol:
            logging.error("Invalid symbol provided")
            return None
        
        data = self.get_historical_data(symbol)
        if data is None or len(data) < self.long_window:
            logging.warning(f"Insufficient historical data for {symbol} (needed: {self.long_window} periods)")
            return None

        # Calculate technical indicators
        data['RSI'] = self.calculate_rsi(data, self.rsi_period)
        data['MACD'], data['Signal'] = self.calculate_macd(data)
        data['Upper'], data['Lower'] = self.calculate_bollinger_bands(data)
        
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility
        
        # Get latest values
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        macd_signal = data['Signal'].iloc[-1]
        upper_band = data['Upper'].iloc[-1]
        lower_band = data['Lower'].iloc[-1]

        # Get news sentiment
        sentiment_data = self.get_news_sentiment(symbol)
        sentiment_strength = 0
        
        if sentiment_data:
            if sentiment_data['sentiment'] == 'positive':
                sentiment_strength = sentiment_data['sentiment_score']
            elif sentiment_data['sentiment'] == 'negative':
                sentiment_strength = sentiment_data['sentiment_score']

        # Trading signals
        signals = {
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'volatility': volatility,
            'sentiment': sentiment_data,
            'recommendation': None,
            'strength': 0,  # 0 to 1 scale
            'reason': []
        }

        # Log current technical indicators
        logging.info(f"\nAnalyzing {symbol}:")
        logging.info(f"Current Price: ${current_price:.2f}")
        logging.info(f"RSI: {rsi:.2f}")
        logging.info(f"MACD: {macd:.4f} (Signal: {macd_signal:.4f})")
        logging.info(f"Volatility: {volatility:.1f}%")
        if sentiment_data:
            logging.info(f"News Sentiment: {sentiment_data['sentiment']} (Score: {sentiment_data['sentiment_score']:.2f})")

        # RSI Analysis
        rsi_strength = 0
        if rsi < 40:
            signals['reason'].append(f"RSI indicates oversold condition ({rsi:.1f})")
            rsi_strength = (40 - rsi) / 40
            logging.info(f"RSI {rsi:.1f} indicates oversold condition - bullish signal")
        elif rsi > 60:
            signals['reason'].append(f"RSI indicates overbought condition ({rsi:.1f})")
            rsi_strength = -(rsi - 60) / 40
            logging.info(f"RSI {rsi:.1f} indicates overbought condition - bearish signal")

        # MACD Analysis
        macd_strength = 0
        if macd > macd_signal:
            signals['reason'].append(f"MACD ({macd:.4f}) above signal line ({macd_signal:.4f})")
            macd_strength = 0.7
            logging.info("MACD above signal line - bullish signal")
        else:
            signals['reason'].append(f"MACD ({macd:.4f}) below signal line ({macd_signal:.4f})")
            macd_strength = -0.7
            logging.info("MACD below signal line - bearish signal")

        # Bollinger Bands Analysis
        bb_strength = 0
        if current_price < lower_band:
            signals['reason'].append(f"Price (${current_price:.2f}) below lower Bollinger Band (${lower_band:.2f})")
            bb_strength = 0.7
            logging.info("Price below lower Bollinger Band - bullish signal")
        elif current_price > upper_band:
            signals['reason'].append(f"Price (${current_price:.2f}) above upper Bollinger Band (${upper_band:.2f})")
            bb_strength = -0.7
            logging.info("Price above upper Bollinger Band - bearish signal")

        # Volatility Analysis
        if volatility > 40:
            signals['reason'].append(f"High volatility ({volatility:.1f}%) - reducing position size")
            logging.info(f"High volatility detected ({volatility:.1f}%) - reducing position size")
        
        # Add sentiment analysis to signals
        if sentiment_data:
            signals['reason'].append(f"News sentiment is {sentiment_data['sentiment']} (score: {sentiment_data['sentiment_score']:.2f})")

        # Combine signals (including sentiment)
        total_strength = (rsi_strength + macd_strength + bb_strength + sentiment_strength) / 4

        if total_strength > 0.15:
            signals['recommendation'] = 'buy'
            signals['strength'] = total_strength
            logging.info(f"Generated BUY signal for {symbol} (strength: {total_strength:.2f})")
        elif total_strength < -0.15:
            signals['recommendation'] = 'short'
            signals['strength'] = abs(total_strength)
            logging.info(f"Generated SHORT signal for {symbol} (strength: {abs(total_strength):.2f})")
        else:
            # Check if we should close positions
            if total_strength > 0 and symbol in self.wallet.short_positions:
                signals['recommendation'] = 'cover'
                signals['strength'] = abs(total_strength)
                logging.info(f"Generated COVER signal for {symbol} (strength: {abs(total_strength):.2f})")
            elif total_strength < 0 and symbol in self.wallet.portfolio:
                signals['recommendation'] = 'sell'
                signals['strength'] = abs(total_strength)
                logging.info(f"Generated SELL signal for {symbol} (strength: {abs(total_strength):.2f})")
            else:
                signals['recommendation'] = 'hold'
                signals['strength'] = abs(total_strength)
                logging.info(f"Generated HOLD signal for {symbol} (strength: {abs(total_strength):.2f})")

        return signals

    def execute_trade(self, symbol):
        """
        Execute trade based on analysis
        Returns: bool indicating if trade was executed
        """
        # Clean symbol
        symbol = self.clean_symbol(symbol)
        if not symbol:
            logging.error("Invalid symbol provided")
            return False
        
        signals = self.analyze_stock(symbol)
        if not signals:
            logging.error(f"Could not analyze {symbol}")
            return False

        if signals['recommendation'] == 'hold':
            logging.info(f"No trade recommended for {symbol}. Reasons: {', '.join(signals['reason'])}")
            return False

        num_shares = self.calculate_position_size(symbol, signals)
        if num_shares == 0:
            logging.info(f"Calculated position size too small for {symbol}")
            return False

        try:
            current_price = self.wallet.get_current_price(symbol)
            trade_reasons = ', '.join(signals['reason'])
            
            if signals['recommendation'] == 'buy':
                success = self.wallet.place_order(symbol, 'buy', num_shares)
                if success:
                    logging.info(f"TRADE: BUY {num_shares} shares of {symbol} @ ${current_price:.2f} | Outcome: NEUTRAL | Reasons: {trade_reasons}")
                return success
            elif signals['recommendation'] == 'sell':
                position = self.wallet.get_position_value(symbol)
                if position:
                    pl_status = "WINNING" if position['unrealized_pl'] > 0 else "LOSING"
                    success = self.wallet.place_order(symbol, 'sell', num_shares)
                    if success:
                        logging.info(f"TRADE: SELL {num_shares} shares of {symbol} @ ${current_price:.2f} | Outcome: {pl_status} (P&L: ${position['unrealized_pl']:.2f}) | Reasons: {trade_reasons}")
                    return success
            elif signals['recommendation'] == 'short':
                success = self.wallet.place_order(symbol, 'short', num_shares)
                if success:
                    logging.info(f"TRADE: SHORT {num_shares} shares of {symbol} @ ${current_price:.2f} | Outcome: NEUTRAL | Reasons: {trade_reasons}")
                return success
            elif signals['recommendation'] == 'cover':
                position = self.wallet.get_position_value(symbol)
                if position:
                    pl_status = "WINNING" if position['unrealized_pl'] > 0 else "LOSING"
                    success = self.wallet.place_order(symbol, 'cover', num_shares)
                    if success:
                        logging.info(f"TRADE: COVER {num_shares} shares of {symbol} @ ${current_price:.2f} | Outcome: {pl_status} (P&L: ${position['unrealized_pl']:.2f}) | Reasons: {trade_reasons}")
                    return success
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}")
            return False

        return False

    def calculate_position_size(self, symbol, signals):
        """Calculate recommended position size based on analysis strength and risk management rules"""
        if signals['recommendation'] == 'hold':
            return 0

        # Get current portfolio value and calculate maximum position size (5% rule - increased from 2%)
        portfolio_summary = self.wallet.get_portfolio_summary()
        total_value = portfolio_summary['total_portfolio_value']
        max_position_size = total_value * 0.05  # 5% maximum per position
        
        # Calculate base position size based on signal strength (1% to 5% of portfolio)
        base_position = total_value * (0.01 + (signals['strength'] * 0.04))
        
        # Adjust position size based on asset type and volatility with less reduction
        if 'volatility' in signals:
            # Reduce position size for high volatility assets, but with a smaller reduction
            volatility_adjustment = 1 - (signals['volatility'] / 200)  # Reduced impact of volatility
            base_position *= max(0.7, volatility_adjustment)  # Don't reduce by more than 30%
        
        # Ensure we don't exceed maximum position size
        position_size = min(base_position, max_position_size)
        
        # Ensure we don't exceed order size limit
        position_size = min(position_size, self.wallet.order_size_limit)
        
        # Calculate number of shares with a minimum of 10 shares
        num_shares = max(10, int(position_size / signals['price']))
        
        # Log position sizing calculation
        logging.info(f"Position sizing for {symbol}:")
        logging.info(f"  Total Portfolio Value: ${total_value:,.2f}")
        logging.info(f"  Maximum Position Size (5%): ${max_position_size:,.2f}")
        logging.info(f"  Calculated Position Size: ${position_size:,.2f}")
        logging.info(f"  Number of Shares: {num_shares}")
        
        return num_shares

    def process_asset(self, asset):
        """
        Process an asset for trading opportunities
        
        Args:
            asset (dict): Asset information containing symbol, current_price, momentum, etc.
        
        Returns:
            bool: True if trade was executed, False otherwise
        """
        try:
            symbol = asset['symbol']
            
            # Execute trading analysis and potential trade
            return self.execute_trade(symbol)
            
        except Exception as e:
            logging.error(f"Error processing asset {asset.get('symbol', 'unknown')}: {str(e)}")
            return False
