import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import pytz
import requests
from requests_cache import CachedSession

class RateLimitManager:
    def __init__(self):
        self.rate_limit_reset_time = None
        self.consecutive_429_count = 0
        self.backoff_multiplier = 1.0

    def handle_rate_limit(self):
        self.consecutive_429_count += 1
        self.backoff_multiplier *= (1.5 ** self.consecutive_429_count)
        wait_time = min(300, 30 * (2 ** (self.consecutive_429_count - 1)))
        self.rate_limit_reset_time = time.time() + wait_time

class MarketData:
    def __init__(self):
        self.session = None
        self.create_new_session()
        self.logger = logging.getLogger(__name__)
        self.monitored_stocks = []
        self.paris_tz = pytz.timezone('Europe/Paris')
        self.rate_limit_manager = RateLimitManager()
        self.delisted_symbols = set()
        self.last_session_refresh = time.time()
        self.session_refresh_interval = 3600
        self.update_monitored_stocks()

    def create_new_session(self):
        if self.session:
            try:
                self.session.close()
            except:
                pass
        self.session = CachedSession(
            'yfinance.cache',
            expire_after=timedelta(minutes=15),
            backend='sqlite',
            serializer='pickle'
        )
        self.last_session_refresh = time.time()

    def refresh_session_if_needed(self):
        if time.time() - self.last_session_refresh > self.session_refresh_interval:
            self.create_new_session()

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def clean_symbol(self, symbol):
        if not symbol:
            return None
        return symbol.replace('$', '').replace(' ', '')

    def get_current_price(self, symbol):
        if symbol in self.delisted_symbols:
            return None

        try:
            self.refresh_session_if_needed()

            ticker = yf.Ticker(symbol, session=self.session)
            hist = ticker.history(period="1d", interval="1h")

            if hist.empty:
                self.delisted_symbols.add(symbol)
                self.log_error(f"${symbol}: possibly delisted; no price data found")
                return None

            return hist['Close'].iloc[-1]

        except Exception as e:
            if "429" in str(e):
                self.rate_limit_manager.handle_rate_limit()
                raise
            elif any(err in str(e).lower() for err in ['timeout', 'connection', 'ssl', 'protocol']):
                self.create_new_session()
                self.log_warning(f"Created new session due to connection error for {symbol}")
                return None
            else:
                self.log_error(f"Unexpected error for {symbol}: {str(e)}")
                return None

    def update_monitored_stocks(self):
        try:
            self.refresh_session_if_needed()
            self.monitored_stocks = []

            # Define the stocks we want to monitor with their respective markets
            market_stocks = {
                'CAC 40': [
                    'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'ATO.PA', 'CS.PA', 'BNP.PA',
                    'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA',
                    'EL.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA', 'MC.PA', 'ML.PA',
                    'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA',
                    'SU.PA', 'STM.PA', 'TEP.PA', 'HO.PA', 'VIE.PA',
                    'DG.PA', 'VIV.PA', 'WLN.PA'
                ],
                'AEX': [
                    'ASML.AS', 'RAND.AS', 'UNA.AS', 'AD.AS', 'INGA.AS', 'PHIA.AS',
                    'ABN.AS', 'KPN.AS', 'HEIA.AS', 'DSM.AS'
                ],
                'BEL 20': [
                    'SOLB.BR', 'KBC.BR', 'UCB.BR', 'ABI.BR', 'PROX.BR',
                    'GLPG.BR', 'GBLB.BR', 'COLR.BR', 'APAM.BR'
                ]
            }

            # Verify each stock and add valid ones to the monitored list
            for market, stocks in market_stocks.items():
                for symbol in stocks:
                    if symbol not in self.delisted_symbols:
                        try:
                            ticker = yf.Ticker(symbol, session=self.session)
                            hist = ticker.history(period="1d")

                            if not hist.empty:
                                self.monitored_stocks.append(symbol)
                                self.log_info(f"Added {symbol} from {market} to monitored stocks")
                            else:
                                self.delisted_symbols.add(symbol)
                                self.log_warning(f"Skipping {symbol} from {market} - no data available")
                                
                        except Exception as e:
                            if "429" in str(e):
                                self.rate_limit_manager.handle_rate_limit()
                                raise
                            self.log_warning(f"Could not validate {symbol} from {market}: {str(e)}")
                            continue

            self.log_info(f"Updated monitored stocks list with {len(self.monitored_stocks)} valid symbols")

        except Exception as e:
            if "429" in str(e):
                self.rate_limit_manager.handle_rate_limit()
                raise
            self.log_error(f"Error updating monitored stocks: {str(e)}")
            self.create_new_session()

    def fetch_stock_data(self, symbol, period="1d", interval="1h"):
        symbol = self.clean_symbol(symbol)
        if not symbol:
            self.logger.error("Invalid symbol provided")
            return None

        try:
            self.refresh_session_if_needed()

            ticker = yf.Ticker(symbol, session=self.session)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError("No data available")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_multiple_stocks_data(self, symbols, period="1d", interval="1d"):
        data = {}
        max_retries = 3

        for symbol in symbols:
            symbol = self.clean_symbol(symbol)
            if not symbol:
                self.logger.error("Invalid symbol provided")
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.refresh_session_if_needed()

                    ticker = yf.Ticker(symbol, session=self.session)
                    df = ticker.history(period=period, interval=interval)

                    if df is not None and not df.empty:
                        data[symbol] = df
                        break

                except Exception as e:
                    retry_count += 1
                    error_msg = str(e).lower()

                    if any(msg in error_msg for msg in ['rate limit', '429', 'too many requests']):
                        self.logger.warning(f"Rate limit hit for {symbol}. Retrying...")
                        time.sleep(2 * (2 ** retry_count))
                    else:
                        self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                        if retry_count == max_retries:
                            break
                        time.sleep(2 * (2 ** retry_count))

        return data

    def get_top_performers(self, limit=50):
        try:
            performers = []
            failed_symbols = []

            batch_size = 10
            for i in range(0, len(self.monitored_stocks), batch_size):
                batch = self.monitored_stocks[i:i+batch_size]
                data = self.get_multiple_stocks_data(batch, period="1d", interval="1d")

                failed_symbols.extend([s for s in batch if s not in data])

                for symbol, df in data.items():
                    if df is not None and not df.empty:
                        try:
                            returns = df['Close'].pct_change().fillna(0)
                            volatility = returns.std()

                            current_price = df['Close'].iloc[-1]
                            avg_volume = df['Volume'].mean()

                            spread = 0.1

                            momentum = returns.tail(10).mean()

                            performers.append({
                                'symbol': symbol,
                                'current_price': current_price,
                                'momentum': momentum,
                                'avg_volume': avg_volume,
                                'spread': spread,
                                'volatility': volatility
                            })

                        except Exception as e:
                            self.logger.error(f"Error processing data for {symbol}: {str(e)}")
                            continue

                time.sleep(0.2)

            if failed_symbols:
                self.logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {', '.join(failed_symbols)}")

            performers = sorted(performers, key=lambda x: x['momentum'], reverse=True)
            return performers[:limit]

        except Exception as e:
            self.logger.error(f"Error getting top performers: {str(e)}")
            return []

    def get_stock_info(self, symbol):
        symbol = self.clean_symbol(symbol)
        if not symbol:
            self.logger.error("Invalid symbol provided")
            return None

        try:
            self.refresh_session_if_needed()

            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            return info

        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {str(e)}")
            return None

    def is_market_open(self):
        try:
            now = datetime.now(self.paris_tz)

            if now.weekday() >= 5:
                return False

            market_open = now.replace(hour=9, minute=0, second=0)
            market_close = now.replace(hour=17, minute=30, second=0)

            return market_open <= now <= market_close

        except Exception as e:
            self.logger.error(f"Error checking market status: {str(e)}")
            return False
