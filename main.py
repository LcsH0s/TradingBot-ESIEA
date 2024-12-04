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
import pandas as pd
from datetime import datetime
import os

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_WALLET_FILE = "wallet.json"

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Trading Bot CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('operation', choices=['run', 'report', 'export'],
                       help='Operation to perform: run (start trading bot), report (show current positions), or export (export to Excel)')
    
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
        return f"\033[92m€{value:,.2f}\033[0m"  # Green for positive
    elif value < 0:
        return f"\033[91m€{value:,.2f}\033[0m"  # Red for negative
    return f"€{value:,.2f}"

def generate_report(wallet: Wallet, logger: logging.Logger) -> None:
    """Generate and display a detailed trading report"""
    initial_balance = wallet.initial_balance  # Get the initial balance
    balance = wallet.balance
    positions = wallet.positions
    trades = wallet.trades
    FEE_RATE = wallet.FEE_RATE  # 0.25%
    
    print("\n=== Trading Bot Report ===\n")
    
    # Calculate total portfolio value and performance
    total_position_value = 0
    total_unrealized_pnl = 0
    
    if positions:
        for ticker, position in positions.items():
            current_price = wallet.get_current_price(ticker, force_update=True)
            position_value = position.quantity * current_price
            total_position_value += position_value
            
            # Calculate cost basis including buy fees
            cost_basis = position.quantity * position.avg_price
            buy_fees = cost_basis * FEE_RATE
            total_cost = cost_basis + buy_fees
            
            # Calculate potential sell fees
            sell_fees = position_value * FEE_RATE
            
            # Unrealized P/L includes both buy and potential sell fees
            unrealized_pnl = position_value - total_cost - sell_fees
            total_unrealized_pnl += unrealized_pnl
    
    total_portfolio_value = balance + total_position_value
    
    # Calculate performance based on initial balance
    total_performance_dollars = total_portfolio_value - initial_balance
    total_performance_pct = (total_performance_dollars / initial_balance) * 100
    
    # Print summary metrics
    print(f"Current Portfolio Value: {format_money(total_portfolio_value)}")
    print(f"Total Performance (%): {'+' if total_performance_pct >= 0 else ''}{total_performance_pct:.2f}%")
    print(f"Total Performance (€): {format_money(total_performance_dollars)}")
    print(f"\nCurrent Balance: {format_money(balance)}")

    # Calculate simulated full sale value
    total_sale_fees = total_position_value * FEE_RATE if total_position_value > 0 else 0
    simulated_total_after_sale = total_portfolio_value - total_sale_fees

    print(f"\nSimulated Portfolio Value After Full Sale:")
    print(f"Total: {format_money(simulated_total_after_sale)} (including {format_money(total_sale_fees)} in selling fees)")
    
    # Current Positions Table
    if positions:
        print("\nCurrent Positions:")
        positions_table = []
        
        for ticker, position in positions.items():
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
            
            # Return percentage calculation including fees
            total_return = (unrealized_pnl / total_cost) * 100 if total_cost != 0 else 0
            
            positions_table.append([
                ticker,
                position.quantity,
                f"€{position.avg_price:.2f}",
                f"€{current_price:.2f}",
                f"€{position_value:.2f}",
                f"€{unrealized_pnl:.2f}",
                f"{total_return:.2f}%"
            ])
        
        print(tabulate(
            positions_table,
            headers=['Ticker', 'Quantity', 'Avg Price (€)', 'Current Price (€)', 'Position Value (€)', 'Unrealized P/L (€)', 'Return'],
            tablefmt='grid'
        ))
    
    # Recent Trades Table
    if trades:
        print("\nRecent Trades:")
        trades_table = []
        total_fees = 0
        
        # Calculate profit/loss for completed trade pairs
        trade_profits = []
        buy_trades = {}
        
        for trade in trades:
            if trade.status == "executed":
                trade_value = trade.price * trade.quantity
                trade_fees = trade_value * FEE_RATE
                total_fees += trade_fees
                
                if trade.type == "buy":
                    # Store buy trade info
                    if trade.ticker not in buy_trades:
                        buy_trades[trade.ticker] = []
                    buy_trades[trade.ticker].append({
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'fees': trade_fees,
                        'date': trade.execution_date
                    })
                elif trade.type == "sell" and trade.ticker in buy_trades:
                    # Match with corresponding buy trade(s)
                    remaining_sell_quantity = trade.quantity
                    sell_value = trade.price * trade.quantity
                    sell_fees = trade_value * FEE_RATE
                    
                    while remaining_sell_quantity > 0 and buy_trades[trade.ticker]:
                        buy_trade = buy_trades[trade.ticker][0]
                        matched_quantity = min(remaining_sell_quantity, buy_trade['quantity'])
                        
                        # Calculate profit for this matched portion
                        buy_cost = matched_quantity * buy_trade['price']
                        sell_revenue = matched_quantity * trade.price
                        
                        # Proportional fees
                        matched_buy_fees = (matched_quantity / buy_trade['quantity']) * buy_trade['fees']
                        matched_sell_fees = (matched_quantity / trade.quantity) * sell_fees
                        
                        profit = sell_revenue - buy_cost - matched_buy_fees - matched_sell_fees
                        
                        trade_profits.append({
                            'ticker': trade.ticker,
                            'profit': profit,
                            'buy_date': buy_trade['date'],
                            'sell_date': trade.execution_date,
                            'quantity': matched_quantity
                        })
                        
                        remaining_sell_quantity -= matched_quantity
                        buy_trade['quantity'] -= matched_quantity
                        
                        if buy_trade['quantity'] <= 0:
                            buy_trades[trade.ticker].pop(0)
                
                trades_table.append([
                    trade.ticker,
                    trade.type.upper(),
                    trade.quantity,
                    f"€{trade.price:.2f}",
                    f"€{trade_value:.2f}",
                    f"€{trade_fees:.2f}",
                    trade.execution_date
                ])
        
        print(tabulate(
            trades_table,
            headers=['Ticker', 'Type', 'Quantity', 'Price', 'Value', 'Fees', 'Execution Date'],
            tablefmt='grid'
        ))
        print(f"\nTotal Trading Fees: €{total_fees:,.2f}")
        
        # Display top 5 most profitable trades
        if trade_profits:
            print("\nTop 5 Most Profitable Trades:")
            top_trades = sorted(trade_profits, key=lambda x: x['profit'], reverse=True)[:5]
            top_trades_table = [
                [
                    trade['ticker'],
                    trade['quantity'],
                    format_money(trade['profit']),
                    trade['buy_date'],
                    trade['sell_date']
                ] for trade in top_trades
            ]
            print(tabulate(
                top_trades_table,
                headers=['Ticker', 'Quantity', 'Profit', 'Buy Date', 'Sell Date'],
                tablefmt='grid'
            ))
    else:
        print("\nNo trades executed yet")
    
def export_to_excel(wallet: Wallet, output_file: Optional[str] = None, logger: logging.Logger = None) -> None:
    """Export wallet data to Excel file with multiple sheets"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"wallet_export_{timestamp}.xlsx"
    
    # Create a Pandas Excel writer
    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    
    # Export Summary
    summary_data = {
        'Metric': ['Initial Balance', 'Current Balance', 'Total Position Value', 'Total Portfolio Value', 'Total Performance (€)', 'Total Performance (%)'],
        'Value': []
    }
    
    # Calculate total position value
    total_position_value = 0
    if wallet.positions:
        for ticker, position in wallet.positions.items():
            current_price = wallet.get_current_price(ticker, force_update=True)
            position_value = position.quantity * current_price
            total_position_value += position_value
    
    total_portfolio_value = wallet.balance + total_position_value
    total_performance_dollars = total_portfolio_value - wallet.initial_balance
    total_performance_pct = (total_performance_dollars / wallet.initial_balance) * 100
    
    summary_data['Value'] = [
        wallet.initial_balance,
        wallet.balance,
        total_position_value,
        total_portfolio_value,
        total_performance_dollars,
        f"{total_performance_pct:.2f}%"
    ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Export Positions
    if wallet.positions:
        positions_data = []
        for ticker, position in wallet.positions.items():
            current_price = wallet.get_current_price(ticker, force_update=True)
            position_value = position.quantity * current_price
            cost_basis = position.quantity * position.avg_price
            unrealized_pnl = position_value - cost_basis
            positions_data.append({
                'Ticker': ticker,
                'Quantity': position.quantity,
                'Average Price': position.avg_price,
                'Current Price': current_price,
                'Position Value': position_value,
                'Cost Basis': cost_basis,
                'Unrealized P/L': unrealized_pnl,
                'Return (%)': (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0
            })
        
        positions_df = pd.DataFrame(positions_data)
        positions_df.to_excel(writer, sheet_name='Positions', index=False)
    
    # Export Trades
    if wallet.trades:
        trades_data = []
        for trade in wallet.trades:
            if trade.status == "executed":
                trades_data.append({
                    'Date': trade.execution_date,
                    'Type': trade.type.upper(),
                    'Ticker': trade.ticker,
                    'Quantity': trade.quantity,
                    'Price': trade.price,
                    'Value': trade.price * trade.quantity
                })
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_excel(writer, sheet_name='Trades', index=False)
    
    # Save the Excel file
    writer.close()
    
    if logger:
        logger.info(f"Wallet data exported to: {output_file}")
    print(f"\nWallet data has been exported to: {output_file}")

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
            
        elif args.operation == 'export':
            main_logger.info("Exporting wallet data to Excel")
            export_to_excel(wallet, logger=logger)
            main_logger.info("Export completed")
            
    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal. Shutting down gracefully...")
        if 'bot' in locals():
            bot.shutdown()
    except Exception as e:
        main_logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
