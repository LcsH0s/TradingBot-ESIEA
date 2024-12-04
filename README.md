# Trading Bot

An advanced algorithmic trading bot that implements multiple technical analysis strategies to make automated trading decisions for stocks listed on major exchanges.

## Features

- **Multiple Trading Strategies**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Combined signal analysis with confidence scoring

- **Portfolio Management**
  - Real-time position tracking
  - Transaction fee consideration (0.25%)
  - Automated trade execution
  - Position sizing and risk management

- **Data Management**
  - Real-time market data from Yahoo Finance
  - Efficient caching system for API calls
  - Historical trade tracking
  - Performance analytics

- **Reporting and Analytics**
  - Real-time portfolio valuation
  - Performance reporting with colored output
  - Excel export functionality
  - Detailed trade history

## Installation

1. Clone the repository:
```bash
git clone git@github.com:LcsH0s/TradingBot-ESIEA.git
cd TradingBot-ESIEA
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a wallet file (optional - will use default if not provided):
```bash
cp wallet.default.json wallet.json
```

## Configuration

1. Edit `config.json` to customize:
   - Trading pairs/tickers
   - Minimum confidence threshold
   - Check interval
   - Technical indicator parameters

2. Environment variables (optional):
   - Create a `.env` file for any API keys or sensitive configuration

## Usage

The bot can be run in different modes:

### Trading Mode
```bash
python main.py run
```

### Generate Reports
```bash
python main.py report
```

### Export Data
```bash
python main.py export
```

### Additional Options
- `-f, --file`: Specify custom wallet file
- `-v, --verbosity`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Docker Support

Build and run using Docker:

```bash
docker-compose up --build
```

Or build manually:

```bash
docker build -t trading-bot .
docker run -v $(pwd)/logs:/app/logs trading-bot
```

## Project Structure

- `main.py`: Entry point and CLI interface
- `trader_bot.py`: Core trading logic and strategy implementation
- `wallet.py`: Portfolio and position management
- `utils/`: Helper utilities and API clients
- `config.json`: Bot configuration
- `wallet.json`: Current portfolio state
- `logs/`: Trading and debug logs

## Technical Indicators

### RSI (Relative Strength Index)
- Period: 14 days
- Overbought level: 70
- Oversold level: 30

### MACD (Moving Average Convergence Divergence)
- Fast period: 12
- Slow period: 26
- Signal period: 9

### Bollinger Bands
- Period: 20
- Standard deviation: 2.0

## Safety Features

- Graceful shutdown handling
- Thread-safe operations
- Cached API calls to prevent rate limiting
- Comprehensive error handling and logging
- Automatic trade execution delay
- Position size limits

## Dependencies

- yfinance >= 0.2.31
- pandas >= 2.1.0
- numpy >= 1.24.0
- tabulate >= 0.9.0
- requests >= 2.31.0
- certifi >= 2023.11.17

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Always perform your own research and risk assessment before trading. The authors are not responsible for any financial losses incurred while using this software.
