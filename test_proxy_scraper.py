import requests
import json
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_endpoint():
    """Test fetching proxies from the API endpoint"""
    try:
        logger.info("Testing API endpoint...")
        response = requests.get('https://api.openproxy.space/http', timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully got response from API")
            
            # Print structure of the response
            logger.info("Response structure:")
            for idx, proxy_list in enumerate(data):
                logger.info(f"\nProxy List {idx + 1}:")
                for key in proxy_list.keys():
                    if key == 'data':
                        logger.info(f"- {key}: {len(proxy_list[key])} proxies")
                    else:
                        logger.info(f"- {key}: {proxy_list[key]}")
                
                # Print first proxy as example
                if proxy_list.get('data'):
                    logger.info("\nExample proxy structure:")
                    logger.info(json.dumps(proxy_list['data'][0], indent=2))
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error testing API endpoint: {str(e)}")

def test_web_scraping():
    """Test scraping proxies directly from the website"""
    try:
        logger.info("\nTesting web scraping...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # First try the list endpoint
        response = requests.get('https://openproxy.space/list/http', headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for script containing proxy data
            scripts = soup.find_all('script')
            nuxt_data = None
            
            for script in scripts:
                content = script.string
                if content and content.startswith('window.__NUXT__='):
                    logger.info("Found Nuxt.js data script")
                    # Extract the JSON-like data
                    try:
                        # Remove the window.__NUXT__= part and any function wrapper
                        data_str = content.split('window.__NUXT__=')[1]
                        if data_str.startswith('(function'):
                            # Find the first { and last }
                            start = data_str.find('{')
                            end = data_str.rfind('}') + 1
                            if start > -1 and end > 0:
                                data_str = data_str[start:end]
                                # Try to parse it as JSON
                                nuxt_data = json.loads(data_str)
                                logger.info("Successfully parsed Nuxt.js data")
                                
                                # Look for proxy data in the parsed object
                                if 'data' in nuxt_data:
                                    logger.info("\nAnalyzing data structure:")
                                    for key in nuxt_data['data']:
                                        logger.info(f"Found data key: {key}")
                                        if isinstance(nuxt_data['data'][key], list):
                                            logger.info(f"List length: {len(nuxt_data['data'][key])}")
                                            if len(nuxt_data['data'][key]) > 0:
                                                logger.info("Sample item structure:")
                                                logger.info(json.dumps(nuxt_data['data'][key][0], indent=2))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON data: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error processing script data: {str(e)}")
            
            if not nuxt_data:
                # Try the API endpoint directly
                logger.info("\nTrying alternative API endpoint...")
                api_response = requests.get('https://openproxy.space/api/lists/http/proxy', headers=headers)
                if api_response.status_code == 200:
                    try:
                        api_data = api_response.json()
                        logger.info("Successfully got API response")
                        logger.info("\nAPI Response structure:")
                        logger.info(json.dumps(api_data[:1], indent=2))
                    except Exception as e:
                        logger.error(f"Error parsing API response: {str(e)}")
                else:
                    logger.error(f"API request failed with status code: {api_response.status_code}")
        else:
            logger.error(f"Web scraping request failed with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error testing web scraping: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting proxy source analysis...")
    test_api_endpoint()
    test_web_scraping()
    logger.info("Analysis complete!")
