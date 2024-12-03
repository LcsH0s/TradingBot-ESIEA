#!/usr/bin/env python3
import argparse
import sys
from wallet import Wallet
from trader_bot import TraderBot
from tabulate import tabulate
from typing import Optional
import json
import logging
from utils.logger import setup_logger, get_class_logger

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_WALLET_FILE = "wallet.json"

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Trading Bot CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('operation', choices=['run', 'report'],
                       help='Operation to perform: run (start trading bot) or report (show current positions)')
    
    parser.add_argument('-f', '--file', type=str, default=DEFAULT_WALLET_FILE,
                       help=f'Wallet file to use (default: {DEFAULT_WALLET_FILE})')
    
    parser.add_argument('-v', '--verbosity', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set the logging verbosity level')
    
    return parser

def load_config() -> dict:
    """Load configuration from config.json if it exists"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'tickers': DEFAULT_TICKERS,
            'min_confidence': 0.7,
            'check_interval': 300  # 5 minutes
        }

def format_money(value: float) -> str:
    """Format money values with colors based on positive/negative"""
    if value > 0:
        return f"\033[92m${value:,.2f}\033[0m"  # Green for positive
    elif value < 0:
        return f"\033[91m${value:,.2f}\033[0m"  # Red for negative
    return f"${value:,.2f}"

def generate_report(wallet: Wallet, logger: logging.Logger) -> None:
    """Generate and display a detailed trading report"""
    balance = wallet.balance
    positions = wallet.positions
    trades = wallet.trades
    FEE_RATE = wallet.FEE_RATE  # 0.25%
    
    print("\n=== Trading Bot Report ===\n")
    print(f"Current Balance: ${balance:,.2f}")
    
    # Current Positions
    if positions:
        print("\nCurrent Positions:")
        positions_table = []
        total_unrealized_pnl = 0
        
        for ticker, position in positions.items():
            # Get latest price from Yahoo Finance
            current_price = wallet.get_current_price(ticker, force_update=True)
            
            # Calculate cost basis including buy fees
            cost_basis = position.quantity * position.avg_price
            buy_fees = cost_basis * FEE_RATE
            total_cost = cost_basis + buy_fees
            
            # Calculate current value and potential sell fees
            position_value = position.quantity * current_price
            sell_fees = position_value * FEE_RATE
            
            # Unrealized P/L includes both buy and potential sell fees
            unrealized_pnl = position_value - total_cost - sell_fees
            total_unrealized_pnl += unrealized_pnl
            
            # Return percentage calculation including fees
            total_return = (unrealized_pnl / total_cost) * 100 if total_cost != 0 else 0
            
            positions_table.append([
                ticker,
                position.quantity,
                f"${position.avg_price:.2f}",
                f"${current_price:.2f}",
                f"${position_value:.2f}",
                f"${unrealized_pnl:.2f}",
                f"{total_return:.2f}%"
            ])
        
        print(tabulate(
            positions_table,
            headers=['Ticker', 'Quantity', 'Avg Price', 'Current Price', 'Position Value', 'Unrealized P/L', 'Return'],
            tablefmt='grid'
        ))
        print(f"\nTotal Unrealized P/L: ${total_unrealized_pnl:,.2f}")
    
    # Recent Trades
    if trades:
        print("\nRecent Trades:")
        trades_table = []
        total_fees = 0
        
        # Dictionary to track realized P/L per ticker
        realized_pnl_by_ticker = {}
        
        for trade in trades:
            if trade.status == "executed":
                trade_value = trade.price * trade.quantity
                trade_fees = trade_value * FEE_RATE
                total_fees += trade_fees
                
                trades_table.append([
                    trade.ticker,
                    trade.type.upper(),
                    trade.quantity,
                    f"${trade.price:.2f}",
                    f"${trade_value:.2f}",
                    f"${trade_fees:.2f}",
                    trade.execution_date
                ])
        
        print(tabulate(
            trades_table,
            headers=['Ticker', 'Type', 'Quantity', 'Price', 'Value', 'Fees', 'Execution Date'],
            tablefmt='grid'
        ))
        # Since we only have buy trades, realized P/L is 0
        print(f"\nTotal Realized P/L: $0.00")
        print(f"Total Trading Fees: ${total_fees:,.2f}")
    else:
        print("\nNo trades executed yet")
    
    # Total P/L (only unrealized since we have no sells)
    total_pnl = total_unrealized_pnl
    print(f"\nTotal P/L (Realized + Unrealized): ${total_pnl:,.2f}")
    
    # Performance Metrics
    if trades:
        executed_trades = [t for t in trades if t.status == "executed"]
        total_trades = len(executed_trades)
        # No winning trades yet since we haven't sold anything
        print(f"\nWin Rate: 0.00% (No completed trades)")
    
def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(level=getattr(logging, args.verbosity))
    main_logger = get_class_logger(logger, "Main")
    
    try:
        # Load configuration
        config = load_config()
        main_logger.debug("Loaded configuration")
        
        # Initialize wallet
        wallet = Wallet(wallet_file=args.file, logger=logger)
        main_logger.info(f"Initialized wallet from file: {args.file}")
        
        if args.operation == 'run':
            main_logger.info(f"Starting trading bot with wallet file: {args.file}")
            main_logger.info(f"Monitoring tickers: {', '.join(config['tickers'])}")
            
            # Initialize and run trading bot
            bot = TraderBot(
                wallet=wallet,
                tickers=config['tickers'],
                min_confidence=config['min_confidence'],
                logger=logger
            )
            bot.run(interval_seconds=config['check_interval'])
            
        elif args.operation == 'report':
            main_logger.info("Generating trading report")
            generate_report(wallet, logger)
            main_logger.info("Report generation completed")
            
    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal. Shutting down gracefully...")
        if 'bot' in locals():
            bot.shutdown()
    except Exception as e:
        main_logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
