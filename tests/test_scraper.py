import unittest
import os
import sys
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrape_google_play_reviews import (
    setup_output_directory,
    verify_app_exists,
    safe_scrape_reviews,
    scrape_bank_reviews,
    BANK_APPS,
    SCRAPING_CONFIG
)

class TestBankReviewScraper(unittest.TestCase):
    """Test cases for the bank review scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_data_dir):
            for file in os.listdir(self.test_data_dir):
                os.remove(os.path.join(self.test_data_dir, file))
            os.rmdir(self.test_data_dir)

    def test_bank_apps_configuration(self):
        """Test if bank apps are properly configured."""
        self.assertIn("Commercial Bank of Ethiopia", BANK_APPS)
        self.assertIn("Bank of Abyssinia", BANK_APPS)
        self.assertIn("Dashen Bank", BANK_APPS)
        
        for bank, info in BANK_APPS.items():
            self.assertIn("package", info)
            self.assertIn("short_name", info)
            self.assertIn("alt_packages", info)

    @patch('scrape_google_play_reviews.app')
    def test_verify_app_exists_success(self, mock_app):
        """Test app verification for existing app."""
        mock_app.return_value = {
            'title': 'CBE Mobile Banking',
            'score': 4.5
        }
        
        exists, info = verify_app_exists('com.cbe.mobilebanking')
        self.assertTrue(exists)
        self.assertEqual(info['title'], 'CBE Mobile Banking')
        self.assertEqual(info['score'], 4.5)

    @patch('scrape_google_play_reviews.app')
    def test_verify_app_exists_failure(self, mock_app):
        """Test app verification for non-existent app."""
        mock_app.side_effect = Exception("App not found")
        exists, info = verify_app_exists('com.nonexistent.app')
        self.assertFalse(exists)
        self.assertIsNone(info)

    @patch('scrape_google_play_reviews.reviews')
    def test_safe_scrape_reviews(self, mock_reviews):
        """Test safe review scraping."""
        test_review = {
            'content': 'Great banking app',
            'score': 5,
            'at': datetime.now()
        }
        mock_reviews.return_value = ([test_review], None)
        
        result = safe_scrape_reviews(
            'com.cbe.mobilebanking',
            'en',
            'et',
            1
        )
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['content'], 'Great banking app')
        self.assertEqual(result[0]['score'], 5)

if __name__ == '__main__':
    unittest.main() 