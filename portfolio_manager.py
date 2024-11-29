#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
from wallet import Wallet
from tabulate import tabulate
import logging

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def generate_portfolio_report(wallet):
    """Generate a detailed portfolio report"""
    summary = wallet.get_portfolio_summary()
    initial_balance = 5000000  # Fixed initial balance of 5M
    
    # Calculate total value and performance
    total_value = summary['total_portfolio_value']  # This now includes all positions correctly
    total_return = total_value - initial_balance
    return_percentage = ((total_value / initial_balance) - 1) * 100
    
    # Portfolio Overview
    print("\n=== Portfolio Overview ===")
    print(f"Initial Balance: {format_currency(initial_balance)}")
    print(f"Available Balance: {format_currency(summary['available_balance'])}")
    print(f"Total Portfolio Value: {format_currency(total_value)}")
    print("=" * 50)
    print(f"Total Return: {format_currency(total_return)} ({return_percentage:+.2f}%)")
    print("=" * 50)
    
    # Long Positions
    if summary['long_positions']:
        print("\n=== Long Positions ===")
        long_data = []
        for symbol, data in summary['long_positions'].items():
            long_data.append([
                symbol,
                data['quantity'],
                format_currency(data['avg_buy_price']),
                format_currency(data['current_price']),
                format_currency(data['current_value']),
                format_currency(data['unrealized_pl']),
                f"{data['unrealized_pl_pct']:+.2f}%"
            ])
        print(tabulate(long_data, 
                      headers=['Symbol', 'Quantity', 'Avg Price', 'Current Price', 'Current Value', 'Unrealized P/L', 'P/L %'],
                      tablefmt='grid'))
    
    # Short Positions
    if summary['short_positions']:
        print("\n=== Short Positions ===")
        short_data = []
        for symbol, data in summary['short_positions'].items():
            short_data.append([
                symbol,
                data['quantity'],
                format_currency(data['avg_short_price']),
                format_currency(data['current_price']),
                format_currency(data['current_value']),
                format_currency(data['unrealized_pl']),
                f"{data['unrealized_pl_pct']:+.2f}%"
            ])
        print(tabulate(short_data,
                      headers=['Symbol', 'Quantity', 'Avg Short Price', 'Current Price', 'Current Value', 'Unrealized P/L', 'P/L %'],
                      tablefmt='grid'))

def sell_all_positions(wallet):
    """Sell all positions and return summary of actions taken"""
    actions_taken = []
    total_pl = 0
    
    # Close long positions
    for symbol in list(wallet.portfolio.keys()):
        position = wallet.get_position_value(symbol)
        if position:
            quantity = wallet.portfolio[symbol]['quantity']
            if wallet.place_order(symbol, 'sell', quantity):
                pl = position['unrealized_pl']
                total_pl += pl
                actions_taken.append({
                    'symbol': symbol,
                    'action': 'SOLD',
                    'quantity': quantity,
                    'pl': pl
                })
    
    # Cover short positions
    for symbol in list(wallet.short_positions.keys()):
        position = wallet.get_position_value(symbol)
        if position:
            quantity = wallet.short_positions[symbol]['quantity']
            if wallet.place_order(symbol, 'cover', quantity):
                pl = position['unrealized_pl']
                total_pl += pl
                actions_taken.append({
                    'symbol': symbol,
                    'action': 'COVERED',
                    'quantity': quantity,
                    'pl': pl
                })
    
    # Print summary of actions
    if actions_taken:
        print("\n=== Position Closing Summary ===")
        action_data = []
        for action in actions_taken:
            action_data.append([
                action['symbol'],
                action['action'],
                action['quantity'],
                format_currency(action['pl'])
            ])
        print(tabulate(action_data,
                      headers=['Symbol', 'Action', 'Quantity', 'P/L'],
                      tablefmt='grid'))
        print(f"\nTotal P/L from closing positions: {format_currency(total_pl)}")
    else:
        print("\nNo positions to close.")

def main():
    parser = argparse.ArgumentParser(description='Portfolio Management Utility')
    parser.add_argument('action', choices=['report', 'sell-all'],
                       help='Action to perform: generate report or sell all positions')
    
    args = parser.parse_args()
    
    try:
        # Initialize wallet
        wallet = Wallet.load_or_create()
        
        if args.action == 'report':
            generate_portfolio_report(wallet)
        elif args.action == 'sell-all':
            print("WARNING: This will sell ALL positions in the portfolio.")
            confirmation = input("Are you sure you want to continue? (yes/no): ")
            if confirmation.lower() == 'yes':
                sell_all_positions(wallet)
                print("\nFinal Portfolio Status:")
                generate_portfolio_report(wallet)
            else:
                print("Operation cancelled.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
