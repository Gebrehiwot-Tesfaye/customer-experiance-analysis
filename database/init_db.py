import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Base, Bank, Review
from database.config import get_database_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('database_init.log')
    ]
)

def init_database():
    """Initialize the database and create tables."""
    try:
        engine = create_engine(get_database_url())
        Base.metadata.create_all(engine)
        return engine
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise

def create_session(engine):
    """Create a new database session."""
    Session = sessionmaker(bind=engine)
    return Session()

def insert_banks(session):
    """Insert bank information into the database."""
    try:
        # First check if banks already exist
        existing_banks = session.query(Bank).all()
        if existing_banks:
            logging.info("Banks already exist in database, skipping insertion")
            return

        banks = [
            Bank(
                name="Commercial Bank of Ethiopia",
                package_name="com.combanketh.mobilebanking",
                short_name="CBE"
            ),
            Bank(
                name="Bank of Abyssinia",
                package_name="com.boa.boaMobileBanking",
                short_name="BOA"
            ),
            Bank(
                name="Dashen Bank",
                package_name="com.dashen.dashensuperapp",
                short_name="DASHEN"
            )
        ]
        
        session.bulk_save_objects(banks)
        session.commit()
        logging.info("Successfully inserted bank information")
    except Exception as e:
        logging.error(f"Error inserting banks: {str(e)}")
        session.rollback()
        raise

def load_reviews(session):
    """Load reviews from CSV and insert into database."""
    try:
        # Get the absolute path to the CSV file
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(current_dir, 'data', 'ethiopian_bank_reviews_20250604_2232.csv')
        
        # Check if file exists
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found at path: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Read CSV file
        logging.info(f"Reading CSV file from: {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Found {len(df)} reviews in CSV file")
        
        # Get bank mappings
        banks = session.query(Bank).all()
        bank_mapping = {bank.name: bank.id for bank in banks}
        logging.info(f"Found banks in database: {list(bank_mapping.keys())}")
        
        # First check if reviews already exist
        existing_reviews = session.query(Review).count()
        if existing_reviews > 0:
            logging.info(f"Found {existing_reviews} existing reviews, deleting them first")
            session.query(Review).delete()
            session.commit()
        
        # Convert reviews to objects
        reviews = []
        for _, row in df.iterrows():
            try:
                # Handle missing or 'Unknown' app version
                app_version = row['app_version'] if pd.notna(row['app_version']) else None
                
                review = Review(
                    review_text=row['review_text'],
                    rating=int(float(row['rating'])),  # Convert to float first in case of decimal ratings
                    review_date=datetime.strptime(row['review_date'], '%Y-%m-%d').date(),
                    source=row['source'],
                    app_version=app_version,
                    thumbs_up_count=int(row['thumbs_up_count']),
                    language=row['language'],
                    country=row['country'],
                    bank_id=bank_mapping[row['bank_name']]
                )
                reviews.append(review)
            except Exception as e:
                logging.warning(f"Error processing review: {str(e)}, Data: {row.to_dict()}")
                continue
        
        # Insert reviews in batches
        batch_size = 100
        total_inserted = 0
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            session.bulk_save_objects(batch)
            session.commit()
            total_inserted += len(batch)
            logging.info(f"Inserted batch of {len(batch)} reviews. Total: {total_inserted}/{len(reviews)}")
        
        logging.info(f"Successfully loaded {total_inserted} reviews into database")
    except Exception as e:
        logging.error(f"Error loading reviews: {str(e)}")
        session.rollback()
        raise

def main():
    """Main function to initialize database and load data."""
    try:
        logging.info("Initializing database...")
        engine = init_database()
        session = create_session(engine)
        
        logging.info("Inserting bank information...")
        insert_banks(session)
        
        logging.info("Loading reviews from CSV...")
        load_reviews(session)
        
        logging.info("✅ Database initialization completed successfully")
    except Exception as e:
        logging.error(f"Error during database initialization: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main() 