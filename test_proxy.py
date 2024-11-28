from market_data import MarketData
import time

def test_market_data():
    print("Testing market data retrieval...")
    start_time = time.time()
    
    market = MarketData()
    
    # Test getting data for a sample stock
    symbol = "AI.PA"  # Airbus stock
    data = market.fetch_stock_data(symbol, period="1d", interval="1h")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nMarket data testing completed in {duration:.2f} seconds")
    if data is not None and not data.empty:
        print(f"Successfully retrieved data for {symbol}")
        print(f"Number of data points: {len(data)}")
        print("\nLatest data point:")
        latest = data.iloc[-1]
        print(f"Open: {latest['Open']:.2f}")
        print(f"High: {latest['High']:.2f}")
        print(f"Low: {latest['Low']:.2f}")
        print(f"Close: {latest['Close']:.2f}")
        print(f"Volume: {latest['Volume']:,}")
    else:
        print(f"Failed to retrieve data for {symbol}")

if __name__ == "__main__":
    test_market_data()
