import ssl
import nltk
import os

def setup_nltk():
    """Setup NLTK data"""
    try:
        # Create SSL context
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Create data directory if it doesn't exist
        if not os.path.exists('nltk_data'):
            os.makedirs('nltk_data')

        # Download required NLTK data
        nltk.download('vader_lexicon', download_dir='nltk_data')
        nltk.download('punkt', download_dir='nltk_data')
        
        print("NLTK setup completed successfully!")
        return True
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")
        return False

if __name__ == "__main__":
    setup_nltk()
