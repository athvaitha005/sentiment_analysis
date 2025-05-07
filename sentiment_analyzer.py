import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import string

# Download required NLTK data
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Define sentiment lexicons
    positive_words = set(['good', 'great', 'amazing', 'excellent', 'fantastic', 'best', 'love', 'happy', 'satisfied', 'perfect', 'awesome', 'wonderful', 'brilliant', 'superb', 'excellent'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'disappointed', 'poor', 'worst', 'hate', 'unhappy', 'unsatisfied', 'useless', 'trash', 'garbage', 'sucks', 'crappy', 'rubbish', 'broken', 'not working', 'doesnt work', 'not good', 'no good', 'issues', 'problem', 'trouble', 'flaw'])
    
    # Split text into words using regex
    words = re.findall(r'\b\w+\b', text)
    
    # Remove stopwords except sentiment words
    stop_words = set(stopwords.words('english'))
    
    # Keep sentiment words and remove other stopwords
    sentiment_words = set(positive_words).union(negative_words)
    words = [word for word in words if word in sentiment_words or word not in stop_words]
    
    # Create n-grams with special handling
    ngrams = []
    for i in range(len(words) - 1):
        # Handle negations
        if words[i] in ['not', 'no', 'never']:
            if words[i+1] in positive_words:
                ngrams.append(f"NEGATED_{words[i+1]}")
                ngrams.append(f"NEGATIVE_CONTEXT")
            elif words[i+1] in negative_words:
                ngrams.append(f"POSITIVE_CONTEXT")
        
        # Handle conjunctions
        if words[i] in ['but', 'however', 'though', 'although']:
            if words[i+1] in negative_words:
                ngrams.append(f"NEGATIVE_CONJUNCTION")
            elif words[i+1] in positive_words:
                ngrams.append(f"POSITIVE_CONJUNCTION")
        
        # Regular n-grams
        ngrams.append(f"{words[i]}_{words[i+1]}")
    
    # Add trigrams
    for i in range(len(words) - 2):
        ngrams.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
    
    # Create sentiment features
    sentiment_features = []
    positive_count = 0
    negative_count = 0
    
    for word in words:
        if word in positive_words:
            positive_count += 1
            sentiment_features.extend(['POSITIVE', 'POSITIVE_INDICATOR'] * 2)
        elif word in negative_words:
            negative_count += 1
            sentiment_features.extend(['NEGATIVE', 'NEGATIVE_INDICATOR'] * 3)
    
    # Add sentiment intensity features
    if positive_count > 0 and negative_count > 0:
        sentiment_features.extend(['MIXED_SENTIMENT'] * 4)  # Increased weight for mixed sentiment
        sentiment_features.extend(['NEGATIVE_CONTEXT'] * 3)
    elif negative_count > 0:
        sentiment_features.extend(['STRONG_NEGATIVE'] * (negative_count * 2))
        sentiment_features.extend(['NEGATIVE_CONTEXT'] * 2)
    
    # Add special cases
    if 'issues' in words or 'problems' in words or 'trouble' in words:
        sentiment_features.extend(['HAS_ISSUES'] * 3)
        sentiment_features.extend(['NEGATIVE_CONTEXT'] * 2)
    
    # Add conjunction context
    if 'but' in words:
        sentiment_features.extend(['CONJUNCTION_CONTEXT'] * 3)
        sentiment_features.extend(['NEGATIVE_CONTEXT'] * 2)
    
    # Combine all features
    all_features = words + ngrams + sentiment_features
    
    return ' '.join(all_features)
    
    # Create n-grams with special handling
    ngrams = []
    for i in range(len(words) - 1):
        # Handle negations
        if words[i] in ['not', 'no', 'never']:
            if words[i+1] in positive_words:
                ngrams.append(f"NEGATED_{words[i+1]}")
                ngrams.append(f"NEGATIVE_CONTEXT")
            elif words[i+1] in negative_words:
                ngrams.append(f"POSITIVE_CONTEXT")
        
        # Handle intensifiers
        if words[i] in ['very', 'extremely', 'totally', 'completely']:
            if words[i+1] in positive_words:
                ngrams.append(f"STRONG_{words[i+1]}")
            elif words[i+1] in negative_words:
                ngrams.append(f"VERY_NEGATIVE")
        
        # Regular n-grams
        ngrams.append(f"{words[i]}_{words[i+1]}")
    
    # Add trigrams
    for i in range(len(words) - 2):
        ngrams.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
    
    # Create sentiment features
    sentiment_features = []
    positive_count = 0
    negative_count = 0
    
    for word in words:
        if word in positive_words:
            positive_count += 1
            sentiment_features.extend(['POSITIVE', 'POSITIVE_INDICATOR'] * 2)
        elif word in negative_words:
            negative_count += 1
            sentiment_features.extend(['NEGATIVE', 'NEGATIVE_INDICATOR'] * 3)
    
    # Add sentiment intensity features
    if positive_count > 0 and negative_count > 0:
        sentiment_features.extend(['MIXED_SENTIMENT'] * 3)
        sentiment_features.extend(['NEGATIVE_CONTEXT'] * 2)
    elif negative_count > 0:
        sentiment_features.extend(['STRONG_NEGATIVE'] * (negative_count * 2))
        sentiment_features.extend(['NEGATIVE_CONTEXT'] * 2)
    
    # Add special cases
    if 'awful' in words or 'terrible' in words or 'horrible' in words:
        sentiment_features.extend(['EXTREMELY_NEGATIVE'] * 4)
    if 'excellent' in words or 'amazing' in words or 'fantastic' in words:
        sentiment_features.extend(['EXTREMELY_POSITIVE'] * 3)
    
    # Combine all features
    all_features = words + ngrams + sentiment_features
    
    return ' '.join(all_features)
    
    # Add n-grams for better context
    ngrams = []
    for i in range(len(words) - 1):
        ngrams.append(f"{words[i]}_{words[i+1]}")
    
    # Combine words and ngrams
    all_terms = words + ngrams
    
    # Add sentiment indicators
    sentiment_indicators = []
    positive_count = 0
    negative_count = 0
    
    for term in all_terms:
        if term in positive_words:
            positive_count += 1
            sentiment_indicators.extend(['POSITIVE', 'POSITIVE_INDICATOR'] * 2)
        elif term in negative_words:
            negative_count += 1
            sentiment_indicators.extend(['NEGATIVE', 'NEGATIVE_INDICATOR'] * 3)
    
    # Add special indicators based on word combinations
    if any(word in words for word in ['not', 'no', 'never']) and any(word in words for word in positive_words):
        sentiment_indicators.extend(['NEGATIVE_CONTEXT'] * 3)
    
    # Combine all terms
    final_terms = all_terms + sentiment_indicators
    
    return ' '.join(final_terms)
    
    # Add sentiment indicators
    sentiment_indicators = []
    
    # Track sentiment counts
    positive_count = 0
    negative_count = 0
    
    for word in words:
        if word in positive_words:
            positive_count += 1
            sentiment_indicators.extend(['POSITIVE', 'POSITIVE_INDICATOR'] * 2)
        elif word in negative_words:
            negative_count += 1
            sentiment_indicators.extend(['NEGATIVE', 'NEGATIVE_INDICATOR'] * 3)
    
    # Add sentiment intensity markers
    if positive_count > 0 and negative_count > 0:
        sentiment_indicators.extend(['MIXED_SENTIMENT'] * 2)
    elif negative_count > 0:
        sentiment_indicators.extend(['STRONG_NEGATIVE'] * (negative_count * 2))
    elif positive_count > 0:
        sentiment_indicators.extend(['STRONG_POSITIVE'] * (positive_count * 2))
    
    # Combine original words with sentiment indicators
    final_words = words + sentiment_indicators
    
    return ' '.join(final_words)

def create_sentiment_model():
    # Load the dataset
    df = pd.read_csv('amazon_review.csv')
    
    # Clean the review text
    df['clean_review'] = df['reviewText'].apply(clean_text)
    
    # Create sentiment labels based on star ratings
    # 1-2 stars: Negative
    # 3 stars: Neutral
    # 4-5 stars: Positive
    df['sentiment'] = df['overall'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))
    
    # Split the data
    X = df['clean_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20000,  # Increased to capture more patterns
            min_df=1,
            max_df=0.99,         # Higher max_df to keep more terms
            stop_words='english',
            lowercase=True,
            analyzer='word',
            token_pattern=r'\b\w+\b',
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            binary=True          # Use binary features
        )),
        ('classifier', MultinomialNB(
            alpha=0.1,           # Reduced alpha for better sensitivity
            fit_prior=True,      # Use prior probabilities
            class_prior=None     # Let the model learn the priors
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = pipeline.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, predictions))
    
    return pipeline

def predict_sentiment(text, model):
    """
    Predict sentiment and star rating for new input text
    Returns: (sentiment, confidence_score, star_rating)
    """
    cleaned_text = clean_text(text)
    sentiment = model.predict([cleaned_text])[0]
    confidence = max(model.predict_proba([cleaned_text])[0])
    
    # Get probability scores for each class
    prob_scores = model.predict_proba([cleaned_text])[0]
    positive_prob = prob_scores[2]  # Index 2 for positive class
    negative_prob = prob_scores[0]  # Index 0 for negative class
    neutral_prob = prob_scores[1]   # Index 1 for neutral class
    
    # Calculate sentiment score
    sentiment_score = negative_prob - positive_prob
    
    # Add special handling for conjunctions
    if 'but' in text.lower() or 'however' in text.lower() or 'though' in text.lower():
        # If there's a conjunction, give more weight to negative context
        if negative_prob > 0.2:  # Lower threshold for conjunction cases
            star_rating = 2 if negative_prob > positive_prob else 3
            sentiment = 'neutral' if sentiment != 'negative' else sentiment
        elif positive_prob > 0.9:  # Very confident positive
            star_rating = 4
        else:
            star_rating = 3
    else:
        # Regular sentiment handling
        if sentiment == 'positive':
            if negative_prob > 0.3:  # Lower threshold for negative
                star_rating = 2 if negative_prob > positive_prob else 3
            elif positive_prob > 0.95:  # Very confident positive
                star_rating = 5
            elif positive_prob > 0.8:  # Moderately confident positive
                star_rating = 4
            else:  # Less confident positive
                star_rating = 3
        elif sentiment == 'neutral':
            if negative_prob > 0.3:  # Lower threshold for negative
                star_rating = 2
            else:
                star_rating = 3
        else:  # negative
            if positive_prob > 0.3:  # Lower threshold for positive
                star_rating = 4 if positive_prob > negative_prob else 3
            elif negative_prob > 0.95:  # Very confident negative
                star_rating = 1
            elif negative_prob > 0.7:  # Moderately confident negative
                star_rating = 2
            else:  # Less confident negative
                star_rating = 3
    
    # Add thresholds for extreme cases
    if sentiment_score > 0.3:  # Lower threshold for negative
        star_rating = min(star_rating, 2)
    elif sentiment_score < -0.3:  # Lower threshold for positive
        star_rating = max(star_rating, 4)
    
    # Special handling for extreme cases
    if 'awful' in text.lower() or 'terrible' in text.lower() or 'horrible' in text.lower():
        star_rating = min(star_rating, 2)
        sentiment = 'negative' if sentiment != 'negative' else sentiment
    if 'excellent' in text.lower() or 'amazing' in text.lower() or 'fantastic' in text.lower():
        star_rating = max(star_rating, 4)
        sentiment = 'positive' if sentiment != 'positive' else sentiment
    
    # Special handling for conjunction cases
    if any(conj in text.lower() for conj in ['but', 'however', 'though', 'although']):
        # If there's a conjunction and any negative words
        if any(neg in text.lower() for neg in ['issues', 'problems', 'trouble', 'flaw']):
            star_rating = min(star_rating, 3)
            sentiment = 'neutral' if sentiment != 'negative' else sentiment
    
    return sentiment, confidence, star_rating

if __name__ == "__main__":
    # Train the model
    model = create_sentiment_model()
    
    print("\nSentiment Analyzer is ready!")
    print("\nType your review (or 'quit' to exit):")
    
    while True:
        review = input("\nYour review: ")
        
        if review.lower() == 'quit':
            print("\nGoodbye!")
            break
            
        sentiment, confidence, stars = predict_sentiment(review, model)
        print(f"\nAnalysis:")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Predicted Stars: {stars}")
