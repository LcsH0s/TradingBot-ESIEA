import json
from datetime import datetime, timedelta
import yfinance as yf
import random
import numpy as np
import pandas as pd

# Configuration
INITIAL_BALANCE = 5_000_000  # $5M initial balance
TARGET_RETURN = 0.15  # 15% return
STOCKS = [
    'NVDA', 'AMD', 'INTC',  # Semiconductors
    'MSFT', 'AAPL', 'GOOGL',  # Tech giants
    'JPM', 'BAC', 'GS',  # Banking
    'PFE', 'JNJ', 'UNH',  # Healthcare
    'XOM', 'CVX', 'COP',  # Energy
    'WMT', 'TGT', 'COST'  # Retail
]

# Use 10 days of data to ensure we get enough trading days
START_DATE = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')  # Extended to 10 days to account for weekends
END_DATE = datetime.now().strftime('%Y-%m-%d')

def find_local_extrema(series, window=2):  # Reduced window size for even more trade opportunities
    maxima = []
    minima = []
    for i in range(window, len(series) - window):
        if all(series[i] >= series[i-j] for j in range(1, window+1)) and \
           all(series[i] >= series[i+j] for j in range(1, window+1)):
            maxima.append(i)
        if all(series[i] <= series[i-j] for j in range(1, window+1)) and \
           all(series[i] <= series[i+j] for j in range(1, window+1)):
            minima.append(i)
    return maxima, minima

def round_datetime(dt):
    """Round datetime to nearest hour"""
    return dt.replace(minute=0, second=0, microsecond=0)

def randomize_time(base_time):
    """Add random minutes and seconds to a base time"""
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    random_microseconds = random.randint(0, 999999)
    return base_time + timedelta(minutes=random_minutes, seconds=random_seconds, microseconds=random_microseconds)

def generate_fake_trades():
    trades = []
    positions = {}
    balance = INITIAL_BALANCE

    # Get historical data for all stocks
    stock_data = {}
    stock_returns = {}
    
    for ticker in STOCKS:
        try:
            stock = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1h')
            if len(stock) < 10:
                continue
                
            stock_data[ticker] = stock
            
            # Calculate returns with some volatility adjustment
            returns = stock['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate simple return using iloc to avoid deprecation warning
            first_price = float(stock['Close'].iloc[0])
            last_price = float(stock['Close'].iloc[-1])
            return_val = (last_price - first_price) / first_price
            stock_returns[ticker] = return_val + volatility * 3  # Increased volatility bonus
            
            print(f"Downloaded {ticker} data: {len(stock)} hours, Return: {return_val:.2%}, Volatility: {volatility:.2%}")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    if not stock_data:
        raise ValueError("No valid stock data found")
    
    # Sort stocks by adjusted return
    sorted_stocks = sorted([(k, float(v)) for k, v in stock_returns.items()], key=lambda x: x[1], reverse=True)
    top_stocks = [stock[0] for stock in sorted_stocks[:10]]  # Keep top 10 stocks
    
    # Start with positions in 6 top stocks
    for ticker in top_stocks[:6]:
        initial_price = float(stock_data[ticker]['Close'].iloc[0])
        initial_quantity = round((0.15 * INITIAL_BALANCE) / initial_price, 2)
        initial_trade_value = initial_quantity * initial_price
        balance -= initial_trade_value
        positions[ticker] = initial_quantity
        
        trades.append({
            "type": "buy",
            "ticker": ticker,
            "quantity": initial_quantity,
            "price": initial_price,
            "date": randomize_time(stock_data[ticker].index[0]).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "status": "executed",
            "execution_date": randomize_time(stock_data[ticker].index[0] + timedelta(seconds=random.randint(30, 300))).strftime("%Y-%m-%dT%H:%M:%S.%f")
        })

    # Generate trades for each stock
    for ticker in top_stocks:
        prices = stock_data[ticker]['Close'].values
        dates = stock_data[ticker].index
        maxima, minima = find_local_extrema(prices)
        
        for idx in sorted(maxima + minima):
            if random.random() < 0.95:
                date = dates[idx]
                price = float(prices[idx])
                
                if idx in maxima and ticker in positions:
                    # Sell at local maximum with improved timing
                    quantity = positions[ticker] * random.uniform(0.6, 1.0)  # More aggressive selling
                    # Add a larger premium to the sell price to simulate excellent timing
                    adjusted_price = price * (1 + random.uniform(0.005, 0.015))  # Increased premium
                    trade_value = quantity * adjusted_price
                    balance += trade_value
                    positions[ticker] -= quantity
                    
                    if positions[ticker] < 0.01:
                        del positions[ticker]
                    
                    trades.append({
                        "type": "sell",
                        "ticker": ticker,
                        "quantity": quantity,
                        "price": adjusted_price,
                        "date": randomize_time(date).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                        "status": "executed",
                        "execution_date": randomize_time(date + timedelta(seconds=random.randint(30, 300))).strftime("%Y-%m-%dT%H:%M:%S.%f")
                    })
                
                elif idx in minima and balance > 0 and len(positions) < 8:  # Limit concurrent positions
                    # Buy at local minimum with improved timing
                    max_investment = min(balance * random.uniform(0.3, 0.5), balance)
                    # Add a larger discount to the buy price to simulate excellent timing
                    adjusted_price = price * (1 - random.uniform(0.005, 0.015))  # Increased discount
                    quantity = round(max_investment / adjusted_price, 2)
                    trade_value = quantity * adjusted_price
                    
                    if trade_value <= balance:
                        balance -= trade_value
                        positions[ticker] = positions.get(ticker, 0) + quantity
                        
                        trades.append({
                            "type": "buy",
                            "ticker": ticker,
                            "quantity": quantity,
                            "price": adjusted_price,
                            "date": randomize_time(date).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                            "status": "executed",
                            "execution_date": randomize_time(date + timedelta(seconds=random.randint(30, 300))).strftime("%Y-%m-%dT%H:%M:%S.%f")
                        })

    # Calculate final portfolio value
    portfolio_value = balance
    for ticker, quantity in positions.items():
        last_price = stock_data[ticker].iloc[-1]['Close'].item()
        portfolio_value += quantity * last_price

    # Create wallet data
    wallet_data = {
        "initial_balance": INITIAL_BALANCE,
        "balance": balance,
        "positions": {},
        "trades": trades
    }

    # Format positions as a dictionary
    for ticker, quantity in positions.items():
        wallet_data["positions"][ticker] = {
            "quantity": float(quantity),
            "avg_price": stock_data[ticker].iloc[-1]['Close'].item()
        }

    # Save to file
    with open('fake_wallet.json', 'w') as f:
        json.dump(wallet_data, f, indent=4)

    print(f"\nInitial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Return: {((portfolio_value/INITIAL_BALANCE - 1) * 100):.2f}%")
    print(f"Number of trades: {len(trades)}")
    
    # Print trade distribution
    trade_counts = {}
    for trade in trades:
        ticker = trade['ticker']
        trade_counts[ticker] = trade_counts.get(ticker, 0) + 1
    print("\nTrades per stock:")
    for ticker, count in sorted(trade_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {count} trades")

if __name__ == "__main__":
    generate_fake_trades()
