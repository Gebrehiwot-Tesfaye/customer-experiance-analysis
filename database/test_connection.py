import psycopg2
from config import DB_CONFIG
import sys

def test_connection():
    """Test PostgreSQL database connection"""
    try:
        # Try to establish a connection
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',  # Try connecting to default database first
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        # Get server version
        cur = conn.cursor()
        cur.execute('SELECT version()')
        version = cur.fetchone()[0]
        print("✅ Successfully connected to PostgreSQL")
        print(f"Server version: {version}")
        
        # Check if our database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['database'],))
        exists = cur.fetchone()
        
        if not exists:
            print(f"Database '{DB_CONFIG['database']}' does not exist.")
            create_db = input(f"Would you like to create database '{DB_CONFIG['database']}'? (y/n): ")
            if create_db.lower() == 'y':
                # Close existing connections first
                cur.close()
                conn.close()
                
                # Connect to default database to create new one
                conn = psycopg2.connect(
                    host=DB_CONFIG['host'],
                    port=DB_CONFIG['port'],
                    database='postgres',
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password']
                )
                conn.autocommit = True
                cur = conn.cursor()
                
                # Create the database
                cur.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
                print(f"✅ Created database '{DB_CONFIG['database']}'")
        else:
            print(f"✅ Database '{DB_CONFIG['database']}' exists")
        
        # Clean up
        cur.close()
        conn.close()
        
    except psycopg2.Error as e:
        print("❌ Error connecting to PostgreSQL database:")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection() 