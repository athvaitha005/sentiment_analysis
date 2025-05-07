from flask import Flask, request, jsonify, render_template
import sentiment_analyzer
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    # Get sentiment analysis results
    sentiment, confidence, stars = sentiment_analyzer.predict_sentiment(text, sentiment_analyzer.model)
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'stars': stars
    })

if __name__ == '__main__':
    app.run(debug=True)
