"""
Bank Reviews Analysis and Insights Generator

This script analyzes the bank reviews data to generate insights, visualizations,
and recommendations for app improvements.

Features:
- Sentiment analysis and trends over time
- Key drivers and pain points identification
- Bank comparison analysis
- Word clouds for common themes
- Rating distributions and trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from collections import Counter
from textblob import TextBlob
import re
from database.config import get_database_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('insights_analysis.log')
    ]
)

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

def connect_to_db():
    """Create database connection"""
    try:
        engine = create_engine(get_database_url())
        return engine
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        raise

def load_data(engine):
    """Load reviews data from database"""
    query = """
    SELECT r.*, b.name as bank_name 
    FROM reviews r
    JOIN banks b ON r.bank_id = b.id
    """
    return pd.read_sql(query, engine)

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0

def extract_keywords(text):
    """Extract important keywords from text"""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Common words to exclude
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    
    # Split into words and remove stop words
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words

def create_rating_distribution(df):
    """Create rating distribution plot comparing banks"""
    plt.figure(figsize=(12, 6))
    
    for bank in df['bank_name'].unique():
        bank_data = df[df['bank_name'] == bank]
        sns.kdeplot(data=bank_data['rating'], label=bank)
    
    plt.title('Rating Distribution by Bank')
    plt.xlabel('Rating')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('visualizations/rating_distribution.png')
    plt.close()

def create_sentiment_trends(df):
    """Create sentiment trends over time"""
    # Add sentiment scores
    df['sentiment'] = df['review_text'].apply(analyze_sentiment)
    
    # Calculate monthly averages
    df['month'] = pd.to_datetime(df['review_date']).dt.to_period('M')
    monthly_sentiment = df.groupby(['month', 'bank_name'])['sentiment'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    monthly_sentiment.plot(marker='o')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.legend(title='Bank')
    plt.grid(True)
    plt.savefig('visualizations/sentiment_trends.png')
    plt.close()

def create_word_clouds(df):
    """Create word clouds for each bank"""
    for bank in df['bank_name'].unique():
        bank_reviews = df[df['bank_name'] == bank]['review_text'].str.cat(sep=' ')
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100
        ).generate(bank_reviews)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Common Words in {bank} Reviews')
        plt.savefig(f'visualizations/wordcloud_{bank.lower().replace(" ", "_")}.png')
        plt.close()

def identify_key_themes(df):
    """Identify key themes and pain points"""
    # Positive keywords indicating good features
    positive_keywords = {
        'fast', 'easy', 'good', 'great', 'excellent', 'convenient', 'helpful',
        'simple', 'reliable', 'secure', 'quick', 'smooth', 'efficient'
    }
    
    # Negative keywords indicating issues
    negative_keywords = {
        'slow', 'crash', 'error', 'bug', 'difficult', 'problem', 'issue',
        'fail', 'poor', 'bad', 'wrong', 'stuck', 'freeze', 'broken'
    }
    
    themes = {'positive': Counter(), 'negative': Counter()}
    
    for _, row in df.iterrows():
        words = extract_keywords(row['review_text'])
        
        # Count positive and negative keywords
        for word in words:
            if word in positive_keywords:
                themes['positive'][word] += 1
            elif word in negative_keywords:
                themes['negative'][word] += 1
    
    return themes

def analyze_bank_comparison(df):
    """Compare banks based on various metrics"""
    comparison = {}
    
    for bank in df['bank_name'].unique():
        bank_data = df[df['bank_name'] == bank]
        
        comparison[bank] = {
            'average_rating': bank_data['rating'].mean(),
            'total_reviews': len(bank_data),
            'sentiment_score': bank_data['review_text'].apply(analyze_sentiment).mean(),
            'recent_trend': bank_data.sort_values('review_date')[-50:]['rating'].mean()
        }
    
    return pd.DataFrame(comparison).round(2)

def generate_insights_report(df, themes, comparison):
    """Generate a comprehensive insights report"""
    report = []
    report.append("=== Bank Mobile Apps Analysis Report ===\n")
    
    # Overall Statistics
    report.append("1. Overall Statistics:")
    report.append(f"- Total reviews analyzed: {len(df)}")
    report.append(f"- Date range: {df['review_date'].min()} to {df['review_date'].max()}")
    report.append(f"- Average rating across all banks: {df['rating'].mean():.2f}\n")
    
    # Bank Comparison
    report.append("2. Bank Comparison:")
    report.append(comparison.to_string())
    report.append("\n")
    
    # Key Drivers (Positive Themes)
    report.append("3. Key Drivers (Most Mentioned Positive Aspects):")
    for word, count in themes['positive'].most_common(5):
        report.append(f"- {word}: mentioned {count} times")
    report.append("")
    
    # Pain Points (Negative Themes)
    report.append("4. Pain Points (Most Mentioned Issues):")
    for word, count in themes['negative'].most_common(5):
        report.append(f"- {word}: mentioned {count} times")
    report.append("")
    
    # Recommendations
    report.append("5. Recommendations:")
    report.append("Based on the analysis, we recommend:")
    
    # Generate recommendations based on the most common issues
    top_issues = [word for word, _ in themes['negative'].most_common(3)]
    recommendations = {
        'crash': "- Improve app stability and implement better error handling",
        'slow': "- Optimize app performance and reduce loading times",
        'error': "- Enhance error messaging and implement automatic retry mechanisms",
        'bug': "- Increase testing coverage and implement automated testing",
        'difficult': "- Simplify user interface and improve user experience"
    }
    
    for issue in top_issues:
        if issue in recommendations:
            report.append(recommendations[issue])
    
    # Ethics and Biases
    report.append("\n6. Notes on Potential Biases:")
    report.append("- Selection bias: Users with extreme experiences (very positive or negative) are more likely to leave reviews")
    report.append("- Timing bias: Recent updates or issues may skew current ratings")
    report.append("- Platform bias: Analysis only includes Google Play Store reviews")
    
    return "\n".join(report)

def main():
    """Main function to run the analysis"""
    try:
        # Create visualizations directory if it doesn't exist
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # Connect to database and load data
        logging.info("Loading data from database...")
        engine = connect_to_db()
        df = load_data(engine)
        
        # Create visualizations
        logging.info("Creating visualizations...")
        create_rating_distribution(df)
        create_sentiment_trends(df)
        create_word_clouds(df)
        
        # Analyze themes and compare banks
        logging.info("Analyzing themes and comparing banks...")
        themes = identify_key_themes(df)
        comparison = analyze_bank_comparison(df)
        
        # Generate and save report
        logging.info("Generating insights report...")
        report = generate_insights_report(df, themes, comparison)
        
        with open('insights_report.txt', 'w') as f:
            f.write(report)
        
        logging.info("Analysis completed successfully!")
        logging.info("Check 'visualizations' directory for plots and 'insights_report.txt' for the detailed report.")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 