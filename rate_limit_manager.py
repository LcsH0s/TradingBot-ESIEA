import time
import logging
from datetime import datetime

class RateLimitManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RateLimitManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self.is_rate_limited = False
        self.rate_limit_reset_time = None
        self.backoff_multiplier = 1.0
        self.consecutive_429_count = 0
        self._initialized = True
    
    def handle_rate_limit(self, reset_time=None):
        """Handle a rate limit being hit"""
        self.is_rate_limited = True
        self.consecutive_429_count += 1
        
        if reset_time:
            self.rate_limit_reset_time = reset_time
        else:
            # If no reset time provided, use exponential backoff
            wait_time = min(300, 30 * (2 ** (self.consecutive_429_count - 1)))
            self.rate_limit_reset_time = time.time() + wait_time
            
        self.backoff_multiplier *= 1.5
        
        self.logger.warning(f"Rate limit hit. Pausing all operations for {wait_time:.1f} seconds")
        return wait_time
    
    def check_and_wait(self):
        """Check if we're rate limited and wait if necessary"""
        if not self.is_rate_limited:
            return False
            
        current_time = time.time()
        if self.rate_limit_reset_time and current_time < self.rate_limit_reset_time:
            wait_time = self.rate_limit_reset_time - current_time
            time.sleep(wait_time)
            
        self.is_rate_limited = False
        self.consecutive_429_count = 0
        self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.5)
        return True
    
    def reset(self):
        """Reset rate limit status after successful operations"""
        self.is_rate_limited = False
        self.consecutive_429_count = 0
        self.backoff_multiplier = 1.0
