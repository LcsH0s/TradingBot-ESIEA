import requests
import random
import logging
from typing import Optional, Dict, List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import socket
import socks
from urllib.parse import urlparse

class ProxyManager:
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.proxies: List[str] = []
        self.working_proxies: List[str] = []
        self.current_proxy_index = 0
        self.last_proxy_refresh = 0
        self.proxy_refresh_interval = 3600  # Refresh proxy list every hour
        self.min_working_proxies = 5  # Minimum number of working proxies to maintain
        self.test_url = 'https://api.binance.com/api/v3/time'  # Test URL
        self.working_proxies_lock = Lock()  # Lock for thread-safe list operations
        self.proxy_url = 'https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt'
        
        # Initialize proxy list
        self._fetch_and_verify_proxies()

    def _fetch_proxies(self) -> None:
        """Fetch SOCKS4 proxies from the provided URL"""
        self.logger.info("Fetching SOCKS4 proxies...")
        try:
            response = requests.get(self.proxy_url, timeout=10)
            if response.status_code == 200:
                # Split the text into lines and remove empty lines
                self.proxies = [proxy.strip() for proxy in response.text.split('\n') if proxy.strip()]
                self.logger.info(f"Successfully fetched {len(self.proxies)} proxies")
            else:
                self.logger.error(f"Failed to fetch proxies. Status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error fetching proxies: {str(e)}")
            self.proxies = []

    def _test_proxy(self, proxy: str) -> Optional[str]:
        """Test a single proxy for functionality"""
        try:
            # Split proxy into host and port
            host, port = proxy.split(':')
            port = int(port)

            # Create a socket and set timeout
            sock = socks.socksocket()
            sock.set_proxy(socks.SOCKS4, host, port)
            sock.settimeout(10)

            # Parse test URL
            parsed = urlparse(self.test_url)
            port = 443 if parsed.scheme == 'https' else 80

            # Connect to test URL
            sock.connect((parsed.hostname, port))
            
            # If we get here, the proxy is working
            self.logger.info(f"Working proxy found: {proxy}")
            return proxy
        except Exception as e:
            return None
        finally:
            try:
                sock.close()
            except:
                pass

    def _verify_proxies(self) -> None:
        """Verify proxies in parallel and keep the working ones"""
        self.logger.info("Starting proxy verification...")
        working_proxies = set()
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_proxy = {executor.submit(self._test_proxy, proxy): proxy 
                             for proxy in self.proxies}
            
            for future in as_completed(future_to_proxy):
                result = future.result()
                if result:
                    working_proxies.add(result)
                    if len(working_proxies) >= self.min_working_proxies:
                        # Cancel remaining futures once we have enough working proxies
                        for f in future_to_proxy:
                            f.cancel()
                        break

        with self.working_proxies_lock:
            self.working_proxies = list(working_proxies)
            self.current_proxy_index = 0
            self.last_proxy_refresh = time.time()
            
        self.logger.info(f"Found {len(self.working_proxies)} working proxies")

    def _fetch_and_verify_proxies(self) -> None:
        """Fetch and verify proxies"""
        self._fetch_proxies()
        self._verify_proxies()

    def get_proxies(self) -> Optional[Dict[str, str]]:
        """Get the next proxy in rotation"""
        with self.working_proxies_lock:
            if not self.working_proxies:
                self._fetch_and_verify_proxies()
                if not self.working_proxies:
                    return None

            # Check if we need to refresh the proxy list
            if time.time() - self.last_proxy_refresh > self.proxy_refresh_interval:
                self._fetch_and_verify_proxies()

            # Rotate through working proxies
            proxy = self.working_proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.working_proxies)

            # Format proxy for requests library
            host, port = proxy.split(':')
            return {
                'http': f'socks4://{host}:{port}',
                'https': f'socks4://{host}:{port}'
            }

    def remove_proxy(self, proxy: str) -> None:
        """Remove a proxy from the rotation if it's not working"""
        if proxy in self.working_proxies:
            self.working_proxies.remove(proxy)
            self.logger.info(f"Removed non-working proxy. {len(self.working_proxies)} working proxies remaining")
            
        # Try to find a new working proxy if we're running low
        if len(self.working_proxies) < self.min_working_proxies:
            self._verify_proxies()
            
    def test_proxy(self, proxy: str) -> bool:
        """Test if a proxy is working"""
        if not proxy:
            return True
            
        try:
            # Split proxy into host and port
            host, port = proxy.split(':')
            port = int(port)

            # Create a socket and set timeout
            sock = socks.socksocket()
            sock.set_proxy(socks.SOCKS4, host, port)
            sock.settimeout(10)

            # Parse test URL
            parsed = urlparse(self.test_url)
            port = 443 if parsed.scheme == 'https' else 80

            # Connect to test URL
            sock.connect((parsed.hostname, port))
            
            # If we get here, the proxy is working
            return True
        except Exception as e:
            return False
        finally:
            try:
                sock.close()
            except:
                pass
