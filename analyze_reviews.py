#!/usr/bin/env python3

"""
Ethiopian Banks Mobile App Review Analysis

This script performs comprehensive sentiment and thematic analysis on bank reviews including:
- Multi-model sentiment analysis (DistilBERT and VADER)
- Keyword extraction and theme clustering
- Visualization of sentiment distributions
- Word clouds for positive and negative reviews
- Combined sentiment-keyword analysis
- Detailed statistical reporting

Features:
- Interactive visualizations using matplotlib and seaborn
- Word clouds for sentiment-specific terms
- Theme identification and clustering
- Cross-model sentiment comparison
- Detailed PDF report generation

Author: Gebrehiwot Tesfaye
Created: 2024-06-04
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import logging
import os
from datetime import datetime
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bank_review_analysis.log'),
        logging.StreamHandler()
    ]
)

class BankReviewAnalyzer:
    """Class to handle all aspects of bank review analysis."""
    
    def __init__(self):
        """Initialize the analyzer with necessary models and configurations."""
        self.setup_visualization_style()
        self.initialize_models()
        
    def setup_visualization_style(self):
        """Configure visualization styles for consistent, professional plots."""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6',
            'primary': '#3498db',
            'secondary': '#f1c40f'
        }
        
    def initialize_models(self):
        """Initialize all required NLP models and resources."""
        logging.info("Initializing NLP models and resources...")
        
        # Download required NLTK resources
        for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logging.error(f"Error downloading NLTK resource {resource}: {str(e)}")
        
        # Initialize sentiment analyzers
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.info("Downloading spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize and remove stopwords
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct 
                 and len(token.lemma_) > 2]
        
        return " ".join(tokens)
    
    def analyze_sentiment(self, text):
        """
        Perform multi-model sentiment analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Combined sentiment analysis results
        """
        try:
            # DistilBERT analysis
            distilbert_result = self.sentiment_analyzer(text)[0]
            
            # VADER analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine consensus sentiment
            vader_sentiment = 'POSITIVE' if vader_scores['compound'] > 0.05 else 'NEGATIVE' if vader_scores['compound'] < -0.05 else 'NEUTRAL'
            
            return {
                'distilbert_label': distilbert_result['label'],
                'distilbert_score': distilbert_result['score'],
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neg': vader_scores['neg'],
                'vader_neu': vader_scores['neu'],
                'vader_sentiment': vader_sentiment,
                'consensus_sentiment': 'POSITIVE' if (distilbert_result['label'] == 'POSITIVE' and vader_scores['compound'] > 0.05) else
                                    'NEGATIVE' if (distilbert_result['label'] == 'NEGATIVE' and vader_scores['compound'] < -0.05) else
                                    'NEUTRAL'
            }
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return None
    
    def extract_themes(self, texts, bank_name=None):
        """
        Extract and analyze themes from reviews.
        
        Args:
            texts (list): List of preprocessed texts
            bank_name (str, optional): Name of the bank for specific analysis
            
        Returns:
            dict: Extracted themes and keywords
        """
        # Initialize TF-IDF with bigrams
        tfidf = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Fit and transform the texts
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        
        # Get top keywords based on TF-IDF scores
        keywords = []
        for idx in range(tfidf_matrix.shape[0]):
            top_indices = tfidf_matrix[idx].toarray()[0].argsort()[-5:][::-1]
            keywords.extend([feature_names[i] for i in top_indices])
        
        # Define theme categories with expanded keywords
        theme_categories = {
            'Account_Access': [
                'login', 'password', 'authentication', 'access', 'account',
                'sign', 'credentials', 'verification', 'security'
            ],
            'Transaction_Performance': [
                'transfer', 'payment', 'transaction', 'speed', 'slow',
                'fast', 'quick', 'delay', 'processing', 'timeout'
            ],
            'UI_Experience': [
                'interface', 'design', 'ui', 'user', 'experience',
                'layout', 'navigation', 'menu', 'screen', 'button'
            ],
            'Customer_Support': [
                'support', 'help', 'service', 'response', 'contact',
                'assistance', 'agent', 'resolution', 'complaint'
            ],
            'Features': [
                'feature', 'functionality', 'option', 'capability', 'service',
                'tool', 'function', 'facility', 'offering'
            ]
        }
        
        # Classify keywords into themes with weights
        theme_distribution = defaultdict(float)
        keyword_themes = defaultdict(list)
        
        for keyword in set(keywords):
            for theme, theme_keywords in theme_categories.items():
                for tk in theme_keywords:
                    if tk in keyword:
                        theme_distribution[theme] += 1
                        keyword_themes[theme].append(keyword)
                        break
        
        return {
            'keywords': list(set(keywords)),
            'themes': dict(theme_distribution),
            'keyword_themes': dict(keyword_themes)
        }
    
    def create_sentiment_visualizations(self, reviews_df, output_dir):
        """
        Create comprehensive sentiment visualization plots.
        
        Args:
            reviews_df (DataFrame): Processed reviews data
            output_dir (str): Directory to save visualizations
        """
        # Create figures directory
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. Overall Sentiment Distribution
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Sentiment distribution by bank
        ax1 = plt.subplot(gs[0, :])
        sentiment_by_bank = pd.crosstab(reviews_df['bank_name'], reviews_df['consensus_sentiment'])
        sentiment_by_bank.plot(kind='bar', stacked=True, ax=ax1,
                             color=[self.colors['positive'], self.colors['neutral'], self.colors['negative']])
        ax1.set_title('Sentiment Distribution by Bank', pad=20)
        ax1.set_xlabel('Bank')
        ax1.set_ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        
        # VADER vs DistilBERT comparison
        ax2 = plt.subplot(gs[1, 0])
        sns.scatterplot(data=reviews_df, x='vader_compound', y='distilbert_score',
                       hue='consensus_sentiment', ax=ax2,
                       palette=[self.colors['negative'], self.colors['neutral'], self.colors['positive']])
        ax2.set_title('VADER vs DistilBERT Sentiment Scores')
        
        # Rating vs Sentiment
        ax3 = plt.subplot(gs[1, 1])
        sns.boxplot(data=reviews_df, x='consensus_sentiment', y='rating', ax=ax3,
                   palette=[self.colors['negative'], self.colors['neutral'], self.colors['positive']])
        ax3.set_title('Rating Distribution by Sentiment')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'sentiment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create word clouds for different sentiments
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            text = ' '.join(reviews_df[reviews_df['consensus_sentiment'] == sentiment]['processed_text'])
            
            if text.strip():
                wordcloud = WordCloud(
                    width=1200, height=800,
                    background_color='white',
                    colormap=LinearSegmentedColormap.from_list('custom', 
                        ['#95a5a6', self.colors['positive' if sentiment == 'POSITIVE' else 'negative']])
                ).generate(text)
                
                plt.figure(figsize=(15, 10))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'{sentiment} Reviews - Word Cloud', pad=20)
                plt.savefig(os.path.join(figures_dir, f'wordcloud_{sentiment.lower()}.png'), 
                          dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Create theme distribution plots
        plt.figure(figsize=(15, 10))
        theme_data = []
        
        for bank in reviews_df['bank_name'].unique():
            bank_reviews = reviews_df[reviews_df['bank_name'] == bank]
            themes = self.extract_themes(bank_reviews['processed_text'].tolist(), bank)
            
            for theme, count in themes['themes'].items():
                theme_data.append({
                    'Bank': bank,
                    'Theme': theme,
                    'Count': count
                })
        
        theme_df = pd.DataFrame(theme_data)
        theme_pivot = theme_df.pivot(index='Bank', columns='Theme', values='Count')
        
        ax = theme_pivot.plot(kind='bar', stacked=True, figsize=(15, 8))
        plt.title('Theme Distribution by Bank', pad=20)
        plt.xlabel('Bank')
        plt.ylabel('Theme Occurrence')
        plt.legend(title='Themes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'theme_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_reviews(self):
        """Main function to analyze bank reviews."""
        logging.info("Starting review analysis...")
        
        # Find most recent review file
        data_dir = "data"
        review_files = [f for f in os.listdir(data_dir) if f.startswith("ethiopian_bank_reviews_")]
        if not review_files:
            logging.error("No review files found!")
            return
        
        latest_file = max(review_files)
        reviews_df = pd.read_csv(os.path.join(data_dir, latest_file))
        
        # Preprocess reviews
        logging.info("Preprocessing reviews...")
        reviews_df['processed_text'] = reviews_df['review_text'].apply(self.preprocess_text)
        
        # Analyze sentiment
        logging.info("Analyzing sentiment...")
        sentiment_results = []
        for text in reviews_df['review_text']:
            result = self.analyze_sentiment(text)
            sentiment_results.append(result if result else {})
        
        # Add sentiment results to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        reviews_df = pd.concat([reviews_df, sentiment_df], axis=1)
        
        # Create visualizations
        logging.info("Creating visualizations...")
        self.create_sentiment_visualizations(reviews_df, data_dir)
        
        # Extract themes and create summary
        logging.info("Analyzing themes...")
        bank_analysis = {}
        
        for bank in reviews_df['bank_name'].unique():
            bank_reviews = reviews_df[reviews_df['bank_name'] == bank]
            themes_result = self.extract_themes(bank_reviews['processed_text'].tolist(), bank)
            
            # Calculate sentiment scores by rating
            sentiment_by_rating = {}
            for rating in range(1, 6):
                rating_reviews = bank_reviews[bank_reviews['rating'] == rating]
                if not rating_reviews.empty:
                    sentiment_by_rating[rating] = {
                        'count': len(rating_reviews),
                        'positive_ratio': (rating_reviews['consensus_sentiment'] == 'POSITIVE').mean(),
                        'negative_ratio': (rating_reviews['consensus_sentiment'] == 'NEGATIVE').mean(),
                        'neutral_ratio': (rating_reviews['consensus_sentiment'] == 'NEUTRAL').mean()
                    }
            
            bank_analysis[bank] = {
                'review_count': len(bank_reviews),
                'average_rating': bank_reviews['rating'].mean(),
                'sentiment_distribution': {
                    'positive': (bank_reviews['consensus_sentiment'] == 'POSITIVE').mean(),
                    'negative': (bank_reviews['consensus_sentiment'] == 'NEGATIVE').mean(),
                    'neutral': (bank_reviews['consensus_sentiment'] == 'NEUTRAL').mean()
                },
                'sentiment_by_rating': sentiment_by_rating,
                'themes': themes_result['themes'],
                'top_keywords_by_theme': themes_result['keyword_themes']
            }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Save detailed results
        reviews_df.to_csv(
            os.path.join(data_dir, f'sentiment_analysis_results_{timestamp}.csv'),
            index=False
        )
        
        # Save aggregated analysis
        with open(os.path.join(data_dir, f'bank_analysis_summary_{timestamp}.json'), 'w') as f:
            json.dump(bank_analysis, f, indent=4)
        
        logging.info("Analysis complete! Results saved to data directory.")
        
        # Print summary
        print("\nAnalysis Summary:")
        print("=" * 80)
        for bank, analysis in bank_analysis.items():
            print(f"\n{bank} Analysis")
            print("-" * 50)
            print(f"Total Reviews: {analysis['review_count']}")
            print(f"Average Rating: {analysis['average_rating']:.2f}")
            
            print("\nSentiment Distribution:")
            print(f"  Positive: {analysis['sentiment_distribution']['positive']:.2%}")
            print(f"  Neutral:  {analysis['sentiment_distribution']['neutral']:.2%}")
            print(f"  Negative: {analysis['sentiment_distribution']['negative']:.2%}")
            
            print("\nTop Themes:")
            for theme, count in sorted(analysis['themes'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {theme}: {count}")
            
            print("\nTop Keywords by Theme:")
            for theme, keywords in analysis['top_keywords_by_theme'].items():
                print(f"  {theme}: {', '.join(keywords[:5])}")
            
            print("\nSentiment by Rating:")
            for rating, stats in analysis['sentiment_by_rating'].items():
                print(f"  {rating} Stars ({stats['count']} reviews):")
                print(f"    Positive: {stats['positive_ratio']:.2%}")
                print(f"    Neutral:  {stats['neutral_ratio']:.2%}")
                print(f"    Negative: {stats['negative_ratio']:.2%}")
            
            print("=" * 80)

if __name__ == "__main__":
    analyzer = BankReviewAnalyzer()
    analyzer.analyze_reviews() 