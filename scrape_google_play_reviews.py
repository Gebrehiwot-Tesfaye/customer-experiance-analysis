#!/usr/bin/env python3

"""
Ethiopian Banks Mobile App Review Scraper

This script scrapes and analyzes customer reviews for Ethiopian bank mobile applications 
from the Google Play Store. It collects reviews for Commercial Bank of Ethiopia (CBE), 
Bank of Abyssinia (BOA), and Dashen Bank.

Features:
- Scrapes reviews, ratings, and dates from Google Play Store
- Handles data preprocessing and cleaning
- Exports data to CSV with descriptive naming
- Includes error handling and logging

Dependencies:
- google-play-scraper: For accessing Google Play Store reviews
- pandas: For data manipulation and CSV export
- datetime: For timestamp handling
- logging: For operation logging

Author: Gebrehiwot Tesfaye
Created: 2025-06-04
Last Modified: 2025-06-09
Version: 1.0.3
"""

import logging
from google_play_scraper import Sort, reviews, app
import pandas as pd
from datetime import datetime
import os
import sys
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bank_review_scraper.log')
    ]
)

# Bank application configurations with correct package names
BANK_APPS = {
    "Commercial Bank of Ethiopia": {
        "package": "com.combanketh.mobilebanking",
        "short_name": "CBE",
        "alt_packages": ["com.cbe.mobilebanking"]
    },
    "Bank of Abyssinia": {
        "package": "com.boa.boaMobileBanking",
        "short_name": "BOA",
        "alt_packages": ["com.boa.apollo"]
    },
    "Dashen Bank": {
        "package": "com.dashen.dashensuperapp",
        "short_name": "DASHEN",
        "alt_packages": ["com.cr2.amolelight"]
    }
}

# Scraping configuration
SCRAPING_CONFIG = {
    "REVIEWS_PER_BANK": 400,
    "SOURCE": "Google Play",
    "LANGUAGES": ["en", "am"],  # Try both English and Amharic
    "COUNTRIES": ["et"],  # Focus on Ethiopia first
    "SORT_METHOD": Sort.MOST_RELEVANT,  # Changed to get most relevant reviews
    "RETRY_DELAY": 2,  # Delay between retries in seconds
    "MAX_RETRIES": 3  # Maximum number of retries per package
}

def generate_output_filename():
    """Generate a descriptive filename for the output CSV."""
    timestamp = datetime.today().strftime('%Y%m%d_%H%M')
    return f"ethiopian_bank_reviews_{timestamp}.csv"

def setup_output_directory():
    """Create 'data' directory if it doesn't exist."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def verify_app_exists(package_name):
    """
    Verify if the app exists on Google Play Store.
    
    Args:
        package_name (str): Google Play Store package name
        
    Returns:
        tuple: (bool, dict) - (True if app exists, app info if exists)
    """
    try:
        app_info = app(package_name)
        logging.info(f"Found app: {app_info['title']} (Rating: {app_info['score']})")
        return True, app_info
    except Exception as e:
        logging.error(f"Error verifying app {package_name}: {str(e)}")
        return False, None

def safe_scrape_reviews(package_name, lang, country, max_reviews):
    """
    Safely scrape reviews with error handling.
    
    Args:
        package_name (str): Package name to scrape
        lang (str): Language code
        country (str): Country code
        max_reviews (int): Maximum number of reviews to fetch
        
    Returns:
        list: List of review data or empty list if failed
    """
    try:
        result, _ = reviews(
            package_name,
            lang=lang,
            country=country,
            sort=SCRAPING_CONFIG["SORT_METHOD"],
            count=max_reviews
        )
        return result
    except Exception as e:
        logging.error(f"Error in safe_scrape_reviews: {str(e)}")
        logging.debug(traceback.format_exc())
        return []

def scrape_bank_reviews(bank_name, bank_info, max_reviews):
    """
    Scrape reviews for a specific bank's mobile app.
    
    Args:
        bank_name (str): Name of the bank
        bank_info (dict): Dictionary containing bank app information
        max_reviews (int): Maximum number of reviews to scrape
        
    Returns:
        list: List of dictionaries containing review data
    """
    all_reviews = []
    packages_to_try = [bank_info["package"]] + bank_info.get("alt_packages", [])
    
    for package_name in packages_to_try:
        app_exists, app_info = verify_app_exists(package_name)
        if not app_exists:
            logging.warning(f"Package {package_name} not found, trying next option...")
            continue
            
        logging.info(f"Attempting to scrape reviews for {bank_name} using package {package_name}")
        
        # Try different combinations of language and country
        for lang in SCRAPING_CONFIG["LANGUAGES"]:
            for country in SCRAPING_CONFIG["COUNTRIES"]:
                retry_count = 0
                while retry_count < SCRAPING_CONFIG["MAX_RETRIES"]:
                    logging.info(f"Attempting to scrape reviews for {bank_name} with lang={lang}, country={country}")
                    
                    result = safe_scrape_reviews(package_name, lang, country, max_reviews)
                    
                    if result:
                        parsed_reviews = [{
                            'review_text': r['content'],
                            'rating': r['score'],
                            'review_date': r['at'].strftime('%Y-%m-%d'),
                            'bank_name': bank_name,
                            'source': SCRAPING_CONFIG["SOURCE"],
                            'app_version': r.get('reviewCreatedVersion', 'Unknown'),
                            'thumbs_up_count': r.get('thumbsUpCount', 0),
                            'language': lang,
                            'country': country,
                            'package_name': package_name
                        } for r in result]
                        
                        all_reviews.extend(parsed_reviews)
                        logging.info(f"Successfully scraped {len(parsed_reviews)} reviews for {bank_name} ({lang}/{country})")
                        break  # Success, exit retry loop
                    
                    retry_count += 1
                    if retry_count < SCRAPING_CONFIG["MAX_RETRIES"]:
                        logging.info(f"Retrying in {SCRAPING_CONFIG['RETRY_DELAY']} seconds... (Attempt {retry_count + 1}/{SCRAPING_CONFIG['MAX_RETRIES']})")
                        time.sleep(SCRAPING_CONFIG["RETRY_DELAY"])
                
                time.sleep(SCRAPING_CONFIG["RETRY_DELAY"])  # Delay between language/country combinations
        
        if all_reviews:  # If we got reviews from this package, no need to try alternatives
            break
    
    return all_reviews

def main():
    """Main execution function for the scraping process."""
    logging.info("Starting the bank review scraping process")
    
    # Setup output directory
    data_dir = setup_output_directory()
    
    # Collect reviews for all banks
    all_reviews = []
    for bank_name, bank_info in BANK_APPS.items():
        bank_reviews = scrape_bank_reviews(
            bank_name,
            bank_info,
            SCRAPING_CONFIG["REVIEWS_PER_BANK"]
        )
        all_reviews.extend(bank_reviews)
    
    # Check if we got any reviews
    if not all_reviews:
        logging.error("No reviews were collected for any bank!")
        return
    
    # Create and process DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Data cleaning and preprocessing
    df.drop_duplicates(subset=['review_text', 'bank_name'], inplace=True)
    df = df.dropna(subset=['review_text', 'rating', 'review_date'])
    df = df.reset_index(drop=True)
    
    # Save to CSV
    output_filename = os.path.join(data_dir, generate_output_filename())
    df.to_csv(output_filename, index=False, encoding='utf-8')
    
    logging.info(f"✅ Successfully scraped and saved {len(df)} reviews to {output_filename}")
    
    # Print summary statistics
    print("\nScraping Summary:")
    print("-" * 50)
    for bank_name in BANK_APPS.keys():
        bank_reviews = df[df['bank_name'] == bank_name]
        if not bank_reviews.empty:
            bank_count = len(bank_reviews)
            avg_rating = bank_reviews['rating'].mean()
            print(f"{bank_name}: {bank_count} reviews, Average rating: {avg_rating:.2f}")
            print("Package distribution:")
            print(bank_reviews['package_name'].value_counts())
            print("Language distribution:")
            print(bank_reviews['language'].value_counts())
            print("-" * 30)
        else:
            print(f"{bank_name}: No reviews collected")

if __name__ == "__main__":
    main()
