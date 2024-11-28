import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import os
import json

class Wallet:
    @staticmethod
    def load_or_create(initial_balance=5000000):
        """
        Static method to load an existing wallet or create a new one
        :param initial_balance: Initial balance for new wallet
        :return: Wallet instance
        """
        if os.path.exists('wallet.json'):
            with open('wallet.json', 'r') as f:
                state = json.load(f)
                initial_balance = state.get('available_balance', initial_balance)
        return Wallet(initial_balance=initial_balance)

    def __init__(self, initial_balance=5000000):
        """Initialize wallet with initial balance and empty portfolio"""
        self.available_balance = initial_balance
        self.initial_balance = initial_balance  # Store initial balance for P&L calculation
        self.portfolio = {}  # Format: {symbol: {'quantity': int, 'avg_buy_price': float}}
        self.short_positions = {}  # Format: {symbol: {'quantity': int, 'avg_short_price': float}}
        self.transaction_fee = 0.0025  # 0.25%
        self.order_size_limit = 1000000  # $1M limit per order
        self.margin_requirement = 0.5  # 50% margin requirement for short positions
        self.trade_history = []  # List to track all trades
        
        # Setup logging
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Configure logging to write to both console and file
        log_filename = f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        # Load existing wallet state if it exists
        if os.path.exists('wallet.json'):
            self.load_state()
        else:
            self.save_state()
            
    def clean_symbol(self, symbol):
        """Clean symbol by removing $ prefix and any whitespace"""
        if not symbol:
            return None
        return symbol.replace('$', '').replace(' ', '')

    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            # Clean symbol
            symbol = self.clean_symbol(symbol)
            if not symbol:
                logging.error("Invalid symbol provided")
                return None
                
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields in order of preference
            price_fields = [
                'regularMarketPrice',
                'currentPrice',
                'previousClose',
                'regularMarketPreviousClose'
            ]
            
            for field in price_fields:
                if field in info and info[field] is not None:
                    return info[field]
            
            # If no price found in info, try getting it from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
                
            logging.error(f"Could not find any valid price for {symbol}")
            return None
            
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {str(e)}")
            return None

    def get_portfolio_value(self):
        """Calculate total portfolio value including available balance"""
        portfolio_value = self.available_balance
        
        # Clean all portfolio symbols
        clean_portfolio = {self.clean_symbol(symbol): data for symbol, data in self.portfolio.items()}
        clean_shorts = {self.clean_symbol(symbol): data for symbol, data in self.short_positions.items()}
        
        for symbol, data in clean_portfolio.items():
            if not symbol:
                continue
            current_price = self.get_current_price(symbol)
            if current_price:
                position_value = data['quantity'] * current_price
                portfolio_value += position_value
        
        for symbol, data in clean_shorts.items():
            if not symbol:
                continue
            current_price = self.get_current_price(symbol)
            if current_price:
                position_value = data['quantity'] * current_price
                portfolio_value += position_value
        
        return portfolio_value

    def get_position_value(self, symbol):
        """Get current value and P&L for a specific position"""
        # Clean symbol
        symbol = self.clean_symbol(symbol)
        if not symbol:
            logging.error("Invalid symbol provided")
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
            
        position_value = 0
        cost_basis = 0
        unrealized_pl = 0
        
        # Long position
        if symbol in self.portfolio:
            position = self.portfolio[symbol]
            position_value = position['quantity'] * current_price
            cost_basis = position['quantity'] * position['avg_buy_price']
            unrealized_pl = position_value - cost_basis
            
        # Short position
        elif symbol in self.short_positions:
            position = self.short_positions[symbol]
            position_value = position['quantity'] * current_price
            cost_basis = position['quantity'] * position['avg_short_price']
            unrealized_pl = cost_basis - position_value  # For shorts, profit is reversed
            
        else:
            return None
        
        return {
            'current_value': position_value,
            'cost_basis': cost_basis,
            'unrealized_pl': unrealized_pl,
            'unrealized_pl_pct': (unrealized_pl / cost_basis) * 100 if cost_basis > 0 else 0
        }

    def place_order(self, symbol, order_type, quantity):
        """
        Place a buy, sell, short sell, or buy to cover order
        order_type: 'buy', 'sell', 'short', or 'cover'
        """
        # Clean symbol
        symbol = self.clean_symbol(symbol)
        if not symbol:
            logging.error("Invalid symbol provided")
            return False
        
        current_price = self.get_current_price(symbol)
        if not current_price:
            logging.error(f"Could not get current price for {symbol}")
            return False

        order_value = quantity * current_price
        fee = order_value * self.transaction_fee

        try:
            if order_type == 'buy':
                if order_value + fee > self.available_balance:
                    logging.error(f"Insufficient funds for buy order: {order_value + fee:.2f} needed")
                    return False
                
                if symbol in self.portfolio:
                    # Update average buy price
                    current_quantity = self.portfolio[symbol]['quantity']
                    current_avg_price = self.portfolio[symbol]['avg_buy_price']
                    new_quantity = current_quantity + quantity
                    new_avg_price = ((current_quantity * current_avg_price) + (quantity * current_price)) / new_quantity
                    self.portfolio[symbol] = {'quantity': new_quantity, 'avg_buy_price': new_avg_price}
                else:
                    self.portfolio[symbol] = {'quantity': quantity, 'avg_buy_price': current_price}
                
                self.available_balance -= (order_value + fee)
                
                # Record the trade
                trade_data = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'value': order_value,
                    'fee': fee
                }
                self.add_trade(trade_data)
                
                logging.info(f"Bought {quantity} shares of {symbol} at ${current_price:.2f}")
                return True

            elif order_type == 'sell':
                if symbol not in self.portfolio or self.portfolio[symbol]['quantity'] < quantity:
                    logging.error(f"Insufficient shares for sell order: {symbol}")
                    return False
                
                # Calculate profit/loss
                avg_buy_price = self.portfolio[symbol]['avg_buy_price']
                profit_loss = (current_price - avg_buy_price) * quantity - fee
                
                # Update portfolio
                self.portfolio[symbol]['quantity'] -= quantity
                if self.portfolio[symbol]['quantity'] == 0:
                    del self.portfolio[symbol]
                
                self.available_balance += (order_value - fee)
                
                # Record the trade
                trade_data = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'sell',
                    'quantity': quantity,
                    'price': current_price,
                    'value': order_value,
                    'fee': fee,
                    'profit_loss': profit_loss
                }
                self.add_trade(trade_data)
                
                logging.info(f"Sold {quantity} shares of {symbol} at ${current_price:.2f}, P&L: ${profit_loss:.2f}")
                return True

            elif order_type == 'short':
                # Check margin requirement
                margin_required = order_value * self.margin_requirement
                if margin_required > self.available_balance:
                    logging.error(f"Insufficient margin for short order: ${margin_required:.2f} needed")
                    return False

                # Simulate order delay
                time.sleep(2)  # 2-second execution delay

                # Execute short order
                self.available_balance -= margin_required  # Reserve margin
                
                if symbol in self.short_positions:
                    # Update average short price
                    current_quantity = self.short_positions[symbol]['quantity']
                    current_avg_price = self.short_positions[symbol]['avg_short_price']
                    total_quantity = current_quantity + quantity
                    new_avg_price = ((current_quantity * current_avg_price) + (quantity * current_price)) / total_quantity
                    
                    self.short_positions[symbol]['quantity'] = total_quantity
                    self.short_positions[symbol]['avg_short_price'] = new_avg_price
                else:
                    self.short_positions[symbol] = {
                        'quantity': quantity,
                        'avg_short_price': current_price
                    }

                logging.info(f"SHORT ORDER: {quantity} {symbol} @ ${current_price:.2f} | Fee: ${fee:.2f}")
                self.save_state()
                return True

            elif order_type == 'cover':
                if symbol not in self.short_positions or self.short_positions[symbol]['quantity'] < quantity:
                    logging.error(f"Insufficient {symbol} short quantity to cover")
                    return False

                # Simulate order delay
                time.sleep(2)  # 2-second execution delay

                # Execute buy to cover order
                self.short_positions[symbol]['quantity'] -= quantity
                cover_cost = order_value + fee
                margin_released = (order_value * self.margin_requirement)
                self.available_balance += margin_released - cover_cost

                # Remove symbol from short positions if quantity is 0
                if self.short_positions[symbol]['quantity'] == 0:
                    del self.short_positions[symbol]

                logging.info(f"COVER ORDER: {quantity} {symbol} @ ${current_price:.2f} | Fee: ${fee:.2f}")
                self.save_state()
                return True

            return False

        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            return False

    def get_portfolio_summary(self):
        """Get a summary of current portfolio positions and their P&L"""
        summary = {
            'available_balance': self.available_balance,
            'long_positions': {},
            'short_positions': {}
        }
        
        total_portfolio_value = self.available_balance
        
        # Clean all portfolio symbols
        clean_portfolio = {self.clean_symbol(symbol): data for symbol, data in self.portfolio.items()}
        clean_shorts = {self.clean_symbol(symbol): data for symbol, data in self.short_positions.items()}
        
        # Long positions
        for symbol, data in clean_portfolio.items():
            if not symbol:
                continue
            position_data = self.get_position_value(symbol)
            if position_data:
                summary['long_positions'][symbol] = {
                    'quantity': data['quantity'],
                    'avg_buy_price': data['avg_buy_price'],
                    **position_data
                }
                total_portfolio_value += position_data['current_value']
        
        # Short positions
        for symbol, data in clean_shorts.items():
            if not symbol:
                continue
            position_data = self.get_position_value(symbol)
            if position_data:
                summary['short_positions'][symbol] = {
                    'quantity': data['quantity'],
                    'avg_short_price': data['avg_short_price'],
                    **position_data
                }
                total_portfolio_value += position_data['unrealized_pl']  # Add unrealized P&L from shorts
        
        summary['total_portfolio_value'] = total_portfolio_value
        return summary

    def load_trade_history(self):
        """Load trade history from wallet state"""
        try:
            if os.path.exists('wallet.json'):
                with open('wallet.json', 'r') as f:
                    state = json.load(f)
                    self.trade_history = state.get('trade_history', [])
                # Convert timestamp strings back to datetime objects
                for trade in self.trade_history:
                    if isinstance(trade['timestamp'], str):
                        trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                # Ensure trades are sorted by date
                self.trade_history.sort(key=lambda x: x['timestamp'])
            else:
                self.trade_history = []
        except Exception as e:
            logging.error(f"Error loading trade history: {str(e)}")
            self.trade_history = []

    def save_trade_history(self):
        """Save trade history as part of wallet state"""
        self.save_state()  # Trade history is now saved as part of the wallet state

    def add_trade(self, trade_data):
        """Add a trade to history and save"""
        trade_copy = trade_data.copy()
        self.trade_history.append(trade_copy)
        # Sort by timestamp to maintain chronological order
        self.trade_history.sort(key=lambda x: x['timestamp'])
        self.save_trade_history()

    def get_trade_performance(self, start_date=None, end_date=None):
        """
        Get performance summary of all trades within the specified date range
        
        Args:
            start_date (datetime, optional): Start date for filtering trades
            end_date (datetime, optional): End date for filtering trades
        
        Returns:
            dict: Performance metrics or None if no trades
        """
        # Calculate unrealized P&L from current positions
        total_unrealized_pl = 0
        for symbol, data in self.portfolio.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                position_value = data['quantity'] * current_price
                cost_basis = data['quantity'] * data['avg_buy_price']
                total_unrealized_pl += position_value - cost_basis
                
        # Calculate realized P&L from completed trades
        filtered_trades = self.trade_history
        if start_date:
            filtered_trades = [t for t in filtered_trades if t['timestamp'] >= start_date]
        if end_date:
            filtered_trades = [t for t in filtered_trades if t['timestamp'] <= end_date]
        
        if not filtered_trades and total_unrealized_pl == 0:
            return None
        
        # Calculate total realized P&L
        total_realized_pl = sum(trade.get('profit_loss', 0) for trade in filtered_trades if 'profit_loss' in trade)
        
        # Total P&L is realized + unrealized
        total_pl = total_realized_pl + total_unrealized_pl
        
        # Sort trades by profit/loss
        completed_trades = [t for t in filtered_trades if 'profit_loss' in t]
        winning_trades = sorted(completed_trades, key=lambda x: x['profit_loss'], reverse=True)
        losing_trades = sorted(completed_trades, key=lambda x: x['profit_loss'])
        
        return {
            'total_trades': len(completed_trades),
            'total_pl': total_pl,
            'realized_pl': total_realized_pl,
            'unrealized_pl': total_unrealized_pl,
            'top_winners': winning_trades[:5],
            'top_losers': losing_trades[:5],
            'win_rate': len([t for t in completed_trades if t['profit_loss'] > 0]) / len(completed_trades) if completed_trades else 0,
            'start_date': min(t['timestamp'] for t in filtered_trades) if filtered_trades else None,
            'end_date': max(t['timestamp'] for t in filtered_trades) if filtered_trades else None
        }

    def save(self):
        """Save wallet state to file"""
        self.save_state()
        
    def save_state(self):
        """Save wallet state to JSON file"""
        try:
            # Prepare trade history for JSON serialization
            history_to_save = []
            for trade in self.trade_history:
                trade_copy = trade.copy()
                if isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                history_to_save.append(trade_copy)

            state = {
                'available_balance': self.available_balance,
                'portfolio': self.portfolio,
                'short_positions': self.short_positions,
                'trade_history': history_to_save
            }
            with open('wallet.json', 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving wallet state: {str(e)}")

    def load_state(self):
        """Load wallet state from JSON file"""
        try:
            with open('wallet.json', 'r') as f:
                state = json.load(f)
            self.available_balance = state.get('available_balance', self.initial_balance)
            self.portfolio = state.get('portfolio', {})
            self.short_positions = state.get('short_positions', {})
            
            # Load and process trade history
            self.trade_history = state.get('trade_history', [])
            for trade in self.trade_history:
                if isinstance(trade['timestamp'], str):
                    trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
            
            # Ensure trades are sorted by date
            self.trade_history.sort(key=lambda x: x['timestamp'])
        except Exception as e:
            logging.error(f"Error loading wallet state: {str(e)}")
            # Initialize with default values if loading fails
            self.available_balance = self.initial_balance
            self.portfolio = {}
            self.short_positions = {}
            self.trade_history = []
