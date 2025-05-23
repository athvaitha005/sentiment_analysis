# Amazon Review Sentiment Analyzer

This project provides a sentiment analysis model for Amazon reviews that can classify new input text as good, bad, or neutral, and predict the corresponding star rating.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the sentiment analyzer:
```bash
python sentiment_analyzer.py
```

The script will:
1. Train a model on your Amazon reviews dataset
2. Display model performance metrics
3. Show example predictions

## Features

- Classifies reviews into three categories:
  - Positive (4-5 stars)
  - Neutral (3 stars)
  - Negative (1-2 stars)
- Provides confidence scores for predictions
- Predicts star ratings based on sentiment
- Includes text preprocessing and cleaning

## Example Output

The script will output model performance metrics and example predictions showing the sentiment, confidence score, and predicted star rating for sample reviews.

## Custom Predictions

You can use the model to make predictions on new text by calling the `predict_sentiment()` function with your input text.
#   S e n t i m e n t   A n a l y s i s  
 