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
    class_logger = get_class_logger(logger, "Report")
    class_logger.info("\n=== Trading Bot Report ===\n")
    
    # Current Balance
    class_logger.info(f"Current Balance: {format_money(wallet.balance)}")
    
    # Current Positions
    positions = []
    total_unrealized_pnl = 0.0
    
    for ticker, position in wallet.positions.items():
        current_price = wallet.get_current_price(ticker)
        position_value = current_price * position.quantity
        unrealized_pnl = position_value - (position.avg_price * position.quantity)
        total_unrealized_pnl += unrealized_pnl
        
        positions.append([
            ticker,
            position.quantity,
            format_money(position.avg_price),
            format_money(current_price),
            format_money(position_value),
            format_money(unrealized_pnl),
            f"{((current_price/position.avg_price - 1) * 100):.2f}%"
        ])
    
    if positions:
        class_logger.info("\nCurrent Positions:")
        class_logger.info("\n" + tabulate(
            positions,
            headers=['Ticker', 'Quantity', 'Avg Price', 'Current Price', 'Position Value', 'Unrealized P/L', 'Return'],
            tablefmt='grid'
        ))
        class_logger.info(f"\nTotal Unrealized P/L: {format_money(total_unrealized_pnl)}")
    else:
        class_logger.info("\nNo current positions")
    
    # Trading History
    realized_pnl = 0.0
    trades = []
    
    for trade in wallet.trades:
        if trade.status == "executed":
            trade_value = trade.price * trade.quantity
            trades.append([
                trade.ticker,
                trade.type.upper(),
                trade.quantity,
                format_money(trade.price),
                format_money(trade_value),
                trade.execution_date
            ])
            
            # Calculate realized P/L
            if trade.type == "sell":
                cost_basis = next(
                    (t.price * t.quantity for t in wallet.trades 
                     if t.ticker == trade.ticker and t.type == "buy" and t.status == "executed"),
                    0
                )
                realized_pnl += trade_value - cost_basis
    
    if trades:
        class_logger.info("\nRecent Trades:")
        class_logger.info("\n" + tabulate(
            trades[-10:],  # Show last 10 trades
            headers=['Ticker', 'Type', 'Quantity', 'Price', 'Value', 'Execution Date'],
            tablefmt='grid'
        ))
        class_logger.info(f"\nTotal Realized P/L: {format_money(realized_pnl)}")
    else:
        class_logger.info("\nNo trades executed yet")
    
    # Total P/L
    total_pnl = realized_pnl + total_unrealized_pnl
    class_logger.info(f"\nTotal P/L (Realized + Unrealized): {format_money(total_pnl)}")
    
    # Performance Metrics
    if trades:
        winning_trades = sum(1 for t in trades if t[4] > 0)
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        class_logger.info(f"\nWin Rate: {win_rate:.2f}%")

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
        wallet = Wallet(wallet_file=args.file)
        main_logger.info(f"Initialized wallet from file: {args.file}")
        
        if args.operation == 'run':
            main_logger.info(f"Starting trading bot with wallet file: {args.file}")
            main_logger.info(f"Monitoring tickers: {', '.join(config['tickers'])}")
            
            # Initialize and run trading bot
            bot = TraderBot(
                wallet=wallet,
                tickers=config['tickers'],
                min_confidence=config['min_confidence']
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
