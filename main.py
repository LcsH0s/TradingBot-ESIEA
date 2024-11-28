import argparse
import time
import logging
import json
from datetime import datetime, timedelta
import signal
import sys
import os
from colorama import init, Fore, Style, Back
from market_data import MarketData
from wallet import Wallet
from trader import Trader
from rate_limit_manager import RateLimitManager

# Initialize colorama with autoreset=True to automatically reset colors
init(autoreset=True)

class TradingBot:
    def __init__(self, num_stocks=50, interval=30, min_volume=500, max_spread=1.0):
        """
        Initialize the trading bot
        :param num_stocks: Number of top performing stocks to monitor
        :param interval: Trading interval in seconds (default: 30 seconds)
        :param min_volume: Minimum average volume for stock selection (default: 500)
        :param max_spread: Maximum bid-ask spread percentage allowed for trading (default: 1.0)
        """
        self.num_stocks = num_stocks
        self.interval = interval
        self.min_volume = min_volume
        self.max_spread = max_spread
        self.start_time = datetime.now()
        self.last_report_time = self.start_time
        self.report_interval = timedelta(minutes=30)  # Generate report every 30 minutes
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Setup logging with both file and console handlers
        self.setup_logging()
        
        # Initialize wallet based on existing state or create new
        if os.path.exists('wallet.json'):
            self.wallet = Wallet()  # Will automatically load state from wallet.json
            self.log_info("Loaded existing wallet state from wallet.json")
        else:
            self.wallet = Wallet(initial_balance=2500000)  # Default initial balance
            self.log_info("Created new wallet with default initial balance")
            
        self.trader = Trader(self.wallet)
        self.market_data = MarketData()
        self.rate_limit_manager = RateLimitManager()
        self.running = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def setup_logging(self):
        """Setup logging configuration with both file and console handlers"""
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(message)s')  # Simpler format for console
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Create new log file with precise timestamp
        start_time = datetime.now()
        log_filename = f'logs/trading_bot_{start_time.strftime("%Y%m%d_%H%M%S")}.log'
        
        # File handler with detailed logging
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Log initial session information
        self.log_info(f"Trading session started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log_info(self, message):
        """Log info message with color"""
        colored_message = f"{Fore.GREEN}{message}{Style.RESET_ALL}"
        self.logger.info(colored_message)

    def log_warning(self, message):
        """Log warning message with color"""
        colored_message = f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        self.logger.warning(colored_message)

    def log_error(self, message):
        """Log error message with color"""
        colored_message = f"{Fore.RED}{message}{Style.RESET_ALL}"
        self.logger.error(colored_message)

    def generate_performance_report(self):
        """Generate a performance report"""
        current_time = datetime.now()
        runtime = current_time - self.start_time
        period = current_time - self.last_report_time
        
        initial_balance = self.wallet.initial_balance
        current_balance = self.wallet.get_portfolio_value()
        total_profit_loss = current_balance - initial_balance
        period_profit_loss = total_profit_loss  # This should be modified to track period-specific P&L
        
        # Color-code the profit/loss values
        total_pl_color = Fore.GREEN if total_profit_loss >= 0 else Fore.RED
        period_pl_color = Fore.GREEN if period_profit_loss >= 0 else Fore.RED
        
        report = f"""
{Fore.CYAN}=== Performance Report ==={Style.RESET_ALL}
Time Period: {period}
Total Runtime: {runtime}
Initial Balance: ${initial_balance:,.2f}
Current Balance: ${current_balance:,.2f}
Total P&L: {total_pl_color}${total_profit_loss:,.2f} ({(total_profit_loss/initial_balance)*100:.2f}%){Style.RESET_ALL}
Period P&L: {period_pl_color}${period_profit_loss:,.2f}{Style.RESET_ALL}
Active Positions: {len(self.wallet.positions)}
"""
        self.log_info(report)
        self.last_report_time = current_time

    def print_market_summary(self):
        """Print market summary"""
        market_summary = self.market_data.get_market_summary()
        if market_summary:
            self.log_info("\nMarket Summary:")
            for market, data in market_summary.items():
                self.log_info(f"{market}: {data['index_price']:.2f} ({data['change']:+.2f}%)")

    def print_status(self):
        """Print current portfolio status"""
        portfolio = self.wallet.get_portfolio_summary()
        self.log_info("\n=== Portfolio Status ===")
        self.log_info(f"Available Balance: €{portfolio['available_balance']:,.2f}")
        self.log_info(f"Total Portfolio Value: €{portfolio['total_portfolio_value']:,.2f}")
        
        if portfolio['positions']:
            self.log_info("\nCurrent Positions:")
            for symbol, data in portfolio['positions'].items():
                self.log_info(f"\n{symbol}:")
                self.log_info(f"  Quantity: {data['quantity']}")
                self.log_info(f"  Avg Buy Price: €{data['avg_buy_price']:.2f}")
                self.log_info(f"  Current Value: €{data['current_value']:,.2f}")
                self.log_info(f"  Unrealized P/L: €{data['unrealized_pl']:,.2f} ({data['unrealized_pl_pct']:.2f}%)")

    def analyze_and_trade(self, asset):
        """Analyze an asset and execute trades if recommended"""
        try:
            # Skip if volume is too low or spread is too high
            if asset['avg_volume'] < self.min_volume:
                self.log_info(f"Skipping {asset['symbol']}: Volume ({asset['avg_volume']:,.0f}) below minimum threshold ({self.min_volume:,.0f})")
                return
            if asset['spread'] > self.max_spread:
                self.log_info(f"Skipping {asset['symbol']}: Spread ({asset['spread']:.2f}%) above maximum threshold ({self.max_spread:.2f}%)")
                return
            
            # Execute trading strategy
            self.trader.process_asset(asset)
            
        except Exception as e:
            self.log_error(f"Error processing {asset['symbol']}: {str(e)}")

    def generate_final_report(self):
        """Generate detailed final performance report"""
        current_time = datetime.now()
        runtime = current_time - self.start_time
        
        # Get basic portfolio information
        initial_balance = self.wallet.initial_balance
        current_balance = self.wallet.get_portfolio_value()
        total_profit_loss = current_balance - initial_balance
        
        # Get trade performance for all time and this session
        all_time_performance = self.wallet.get_trade_performance()
        session_performance = self.wallet.get_trade_performance(start_date=self.start_time, end_date=current_time)
        
        if all_time_performance:
            all_time_win_rate = all_time_performance['win_rate'] * 100
            all_time_trades = all_time_performance['total_trades']
            all_time_pl = all_time_performance['total_pl']
            realized_pl = all_time_performance['realized_pl']
            unrealized_pl = all_time_performance['unrealized_pl']
            
            session_win_rate = session_performance['win_rate'] * 100 if session_performance else 0
            session_trades = session_performance['total_trades'] if session_performance else 0
            session_pl = session_performance['total_pl'] if session_performance else 0
            
            report = f"""
{Fore.CYAN}========== Final Trading Session Report =========={Style.RESET_ALL}
Session Runtime: {runtime}
Session Trades: {session_trades}
Session P&L: {Fore.GREEN if session_pl >= 0 else Fore.RED}${session_pl:,.2f}{Style.RESET_ALL}
Session Win Rate: {Fore.GREEN if session_win_rate >= 50 else Fore.RED}{session_win_rate:.1f}%{Style.RESET_ALL}

{Fore.CYAN}All-Time Performance:{Style.RESET_ALL}
Trading Period: {all_time_performance['start_date'].strftime('%Y-%m-%d') if all_time_performance['start_date'] else 'N/A'} to {all_time_performance['end_date'].strftime('%Y-%m-%d') if all_time_performance['end_date'] else 'N/A'}
Total Trades: {all_time_trades}
Initial Balance: ${initial_balance:,.2f}
Current Balance: ${current_balance:,.2f}
Total P&L: {Fore.GREEN if all_time_pl >= 0 else Fore.RED}${all_time_pl:,.2f} ({(all_time_pl/initial_balance)*100:.2f}%){Style.RESET_ALL}
Realized P&L: {Fore.GREEN if realized_pl >= 0 else Fore.RED}${realized_pl:,.2f}{Style.RESET_ALL}
Unrealized P&L: {Fore.GREEN if unrealized_pl >= 0 else Fore.RED}${unrealized_pl:,.2f}{Style.RESET_ALL}
Win Rate: {Fore.GREEN if all_time_win_rate >= 50 else Fore.RED}{all_time_win_rate:.1f}%{Style.RESET_ALL}

{Fore.GREEN}Top 5 All-Time Winning Trades:{Style.RESET_ALL}"""
            
            # Add top winning trades
            for trade in all_time_performance['top_winners']:
                trade_date = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                report += f"\n{trade_date} | {trade['symbol']}: ${trade['profit_loss']:,.2f} | {trade['quantity']} shares @ ${trade['price']:.2f}"
            
            report += f"\n\n{Fore.RED}Top 5 All-Time Losing Trades:{Style.RESET_ALL}"
            
            # Add top losing trades
            for trade in all_time_performance['top_losers']:
                trade_date = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                report += f"\n{trade_date} | {trade['symbol']}: ${trade['profit_loss']:,.2f} | {trade['quantity']} shares @ ${trade['price']:.2f}"
            
            report += f"\n\n{Fore.CYAN}==========================================={Style.RESET_ALL}"
            
            self.log_info(report)
        else:
            self.log_info(f"""
{Fore.CYAN}========== Final Trading Session Report =========={Style.RESET_ALL}
Runtime: {runtime}
Initial Balance: ${initial_balance:,.2f}
Final Balance: ${current_balance:,.2f}
Total P&L: {Fore.GREEN if total_profit_loss >= 0 else Fore.RED}${total_profit_loss:,.2f} ({(total_profit_loss/initial_balance)*100:.2f}%){Style.RESET_ALL}

No trades have been executed yet.
{Fore.CYAN}==========================================={Style.RESET_ALL}
""")

    def run(self):
        """Run the trading bot"""
        self.running = True
        self.log_info(f"Starting trading bot with {self.num_stocks} stocks, {self.interval}s interval")
        
        try:
            while self.running:
                try:
                    # Check for rate limiting before proceeding
                    if self.rate_limit_manager.check_and_wait():
                        self.log_warning("Resuming operations after rate limit pause")
                    
                    # Update monitored stocks list periodically
                    self.market_data.update_monitored_stocks()
                    
                    # Process each monitored stock
                    for symbol in self.market_data.monitored_stocks:
                        if not self.running:
                            break
                            
                        try:
                            # Get current market data
                            current_price = self.market_data.get_current_price(symbol)
                            if current_price is None:
                                if self.rate_limit_manager.is_rate_limited:
                                    self.log_warning(f"Rate limit hit while processing {symbol}, pausing operations")
                                    break
                                continue
                                
                            # Process the asset
                            self.analyze_and_trade({
                                'symbol': symbol,
                                'price': current_price,
                                'avg_volume': 1000,  # This should be fetched from market data
                                'spread': 0.5  # This should be calculated from bid/ask
                            })
                            
                        except Exception as e:
                            if "429" in str(e) or "rate limit" in str(e).lower():
                                self.rate_limit_manager.handle_rate_limit()
                                self.log_warning(f"Rate limit hit while processing {symbol}, pausing operations")
                                break
                            else:
                                self.log_error(f"Error processing {symbol}: {str(e)}")
                    
                    # Generate performance report if interval has elapsed
                    current_time = datetime.now()
                    if current_time - self.last_report_time >= self.report_interval:
                        self.generate_performance_report()
                    
                    # Sleep for the trading interval if we're not rate limited
                    if not self.rate_limit_manager.is_rate_limited:
                        time.sleep(self.interval)
                        
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        self.rate_limit_manager.handle_rate_limit()
                        self.log_warning("Rate limit hit in main loop, pausing operations")
                    else:
                        self.log_error(f"Error in main loop: {str(e)}")
                        time.sleep(self.interval)  # Sleep to prevent rapid error loops
                        
        except KeyboardInterrupt:
            self.log_info("Received keyboard interrupt, shutting down...")
        finally:
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        """Gracefully shutdown the bot"""
        self.log_info("\nReceived shutdown signal. Stopping trading bot...")
        self.running = False
        
        # Generate final performance report
        self.log_info("\nGenerating final performance report...")
        self.generate_final_report()
        
        # Save final wallet state
        self.wallet.save_state()
        self.log_info("\nWallet state saved.")
        
        self.log_info("\nTrading bot shutdown complete. Goodbye!")
        sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description='High-Frequency European Stock Trading Bot')
    parser.add_argument(
        '--num-stocks',
        type=int,
        default=50,
        help='Number of top performing stocks to analyze'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Trading interval in seconds'
    )
    parser.add_argument(
        '--min-volume',
        type=int,
        default=500,
        help='Minimum average volume for stock selection'
    )
    parser.add_argument(
        '--max-spread',
        type=float,
        default=1.0,
        help='Maximum bid-ask spread percentage allowed for trading'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate intervals
    if args.interval < 5:
        print(f"{Fore.YELLOW}Interval cannot be less than 5 seconds due to exchange rate limits. Setting to 5 seconds.{Style.RESET_ALL}")
        args.interval = 5
    
    # Initialize bot
    bot = TradingBot(
        num_stocks=args.num_stocks,
        interval=args.interval,
        min_volume=args.min_volume,
        max_spread=args.max_spread
    )
    
    # Display configuration
    print("\nHFT Bot Configuration:")
    print(f"- Monitoring top {args.num_stocks} performing assets")
    print(f"- Trading every {args.interval} seconds")
    print(f"- Minimum volume: {args.min_volume:,} shares")
    print(f"- Maximum spread: {args.max_spread}%")
    print(f"\nInitial Portfolio Value: €{bot.wallet.get_portfolio_value():,.2f}")
    
    print("\nStarting HFT bot...")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, bot.shutdown)
    signal.signal(signal.SIGTERM, bot.shutdown)
    
    # Run the bot
    bot.run()

if __name__ == "__main__":
    main()
