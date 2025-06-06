import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),  # Default PostgreSQL port
    'database': os.getenv('DB_NAME', 'bank_reviews'),
    'user': os.getenv('DB_USER', 'postgres'),  # Default PostgreSQL superuser
    'password': os.getenv('DB_PASSWORD', '')  # Should be set in .env file
}

# SQLAlchemy connection string
def get_database_url():
    # URL encode the password to handle special characters
    password = quote_plus(DB_CONFIG['password'])
    return f"postgresql://{DB_CONFIG['user']}:{password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}" 