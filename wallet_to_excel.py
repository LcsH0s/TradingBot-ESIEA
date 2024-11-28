import json
import pandas as pd
import os
from datetime import datetime

def wallet_to_excel():
    # Check if exported.json exists
    if not os.path.exists('wallet.json'):
        print("wallet.json not found. Please run the trading bot first to generate the wallet file.")
        return

    # Read the exported.json file
    with open('wallet.json', 'r') as f:
        wallet_data = json.load(f)

    # Create DataFrames for different sections
    # Main wallet info
    wallet_info = pd.DataFrame({
        'Metric': ['Available Balance', 'Initial Balance'],
        'Value': [wallet_data.get('available_balance', 0), wallet_data.get('initial_balance', 0)]
    })

    # Portfolio positions
    portfolio_data = []
    for symbol, data in wallet_data.get('portfolio', {}).items():
        portfolio_data.append({
            'Symbol': symbol,
            'Quantity': data['quantity'],
            'Average Buy Price': data['avg_buy_price']
        })
    portfolio_df = pd.DataFrame(portfolio_data) if portfolio_data else pd.DataFrame()

    # Short positions
    short_data = []
    for symbol, data in wallet_data.get('short_positions', {}).items():
        short_data.append({
            'Symbol': symbol,
            'Quantity': data['quantity'],
            'Average Short Price': data['avg_short_price']
        })
    short_df = pd.DataFrame(short_data) if short_data else pd.DataFrame()

    # Trade history
    trade_history_df = pd.DataFrame(wallet_data.get('trade_history', []))

    # Create Excel writer object
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f'wallet_export_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename) as writer:
        # Write each DataFrame to a different sheet
        wallet_info.to_excel(writer, sheet_name='Wallet Info', index=False)
        if not portfolio_df.empty:
            portfolio_df.to_excel(writer, sheet_name='Portfolio', index=False)
        if not short_df.empty:
            short_df.to_excel(writer, sheet_name='Short Positions', index=False)
        if not trade_history_df.empty:
            trade_history_df.to_excel(writer, sheet_name='Trade History', index=False)

    print(f"Excel file created successfully: {excel_filename}")

if __name__ == "__main__":
    wallet_to_excel()
