#!/usr/bin/env python3
import argparse
from utils.stock_screener import StockScreener
from utils.logger import setup_logger
import sys

def main():
    parser = argparse.ArgumentParser(description='Update trading bot config with top performing S&P 500 stocks')
    parser.add_argument('-n', '--number', type=int, default=20,
                       help='Number of top performers to select (default: 20)')
    parser.add_argument('-d', '--days', type=int, default=30,
                       help='Lookback period in days (default: 30)')
    parser.add_argument('-c', '--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    logger = setup_logger('update_tickers')
    
    try:
        logger.info(f"Starting stock screening (top {args.number} performers, {args.days} days lookback)")
        screener = StockScreener(lookback_period=args.days)
        screener.update_config(config_path=args.config, top_n=args.number)
        logger.info("Stock screening complete")
        
    except Exception as e:
        logger.error(f"Error updating tickers: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
