import requests
import json
import logging
from datetime import datetime, timedelta
import os
from transformers import pipeline
from google import genai
from google.genai import types

class NewsAgent:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        with open(config_path) as f:
            self.keys = json.load(f)
        self.api_key = self.keys.get('newsapi_key', '')
        self.gemini_key = self.keys.get('gemini_api_key', '')

        # Load FinBERT sentiment model
        self.sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

        # Initialize Gemini model
        self.gemini_model = None
        if self.gemini_key:
            try:
                client = genai.Client(
                    vertexai=True, 
                    project='nimble-thinker-472714-r0', 
                    location='us-central1'
                )
                # Store the model properly
                self.gemini_model = client.models.generate_content
                self.client = client
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
                self.gemini_model = None

        # Set up logging
        self.logger = logging.getLogger('NewsAgent')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Cache for news articles (key: symbol_days -> {'articles': [...], 'timestamp': datetime})
        self.news_cache = {}
        self.cache_duration = 900  # 15 minutes in seconds

    def fetch_news(self, symbol, days=10):
        """
        Fetch news articles for a given symbol from the last 'days' days.
        Uses caching to avoid repeated API calls.
        """
        if not self.api_key:
            return {"error": "NewsAPI key not configured"}

        cache_key = f"{symbol}_{days}"
        now = datetime.now()

        # Check cache
        if cache_key in self.news_cache:
            cached_data = self.news_cache[cache_key]
            if (now - cached_data['timestamp']).total_seconds() < self.cache_duration:
                self.logger.info(f"Cache hit for {cache_key}, returning {len(cached_data['articles'])} cached articles")
                return cached_data['articles']
            else:
                self.logger.info(f"Cache expired for {cache_key}, fetching fresh data")
        else:
            self.logger.info(f"Cache miss for {cache_key}, fetching from API")

        from_date = (now - timedelta(days=days)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&sortBy=publishedAt&apiKey={self.api_key}&language=en"

        try:
            self.logger.info(f"Fetching news from NewsAPI for {symbol}, days={days}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            self.logger.info(f"Retrieved {len(articles)} articles from NewsAPI")

            # Cache the result
            self.news_cache[cache_key] = {
                'articles': articles,
                'timestamp': now
            }
            self.logger.info(f"Cached {len(articles)} articles for key {cache_key}")

            return articles
        except requests.RequestException as e:
            self.logger.error(f"NewsAPI request failed: {str(e)}")
            return {"error": str(e)}

    def analyze_sentiment(self, articles):
        """
        Analyze sentiment for each article using FinBERT.
        Returns list of articles with sentiment scores.
        """
        if isinstance(articles, dict) and 'error' in articles:
            return articles

        self.logger.info(f"Analyzing sentiment for {len(articles)} articles")
        tagged_articles = []
        for i, art in enumerate(articles):
            text = art.get('title', '') + ' ' + art.get('description', '')
            if text.strip():
                result = self.sentiment_model(text[:512])  # FinBERT has 512 token limit
                sentiment = result[0]['label'].lower()
                score = result[0]['score']
                tagged_articles.append({
                    "title": art.get('title', ''),
                    "source": art.get('source', {}).get('name', ''),
                    "published_at": art.get('publishedAt', ''),  # Fixed: was 'published_at'
                    "model_sentiment": sentiment,
                    "model_score": score,
                    "text": text
                })
                if i < 3:  # Log first few for debugging
                    self.logger.debug(f"Article {i+1}: {sentiment} ({score:.2f}) - {art.get('title', '')[:50]}...")
        self.logger.info(f"Completed sentiment analysis for {len(tagged_articles)} articles")
        return tagged_articles

    def summarize_with_gemini(self, tagged_articles, symbol, max_articles=5):
        """
        Use Gemini to summarize and reason about the news sentiment.
        """
        if not self.gemini_model:
            return {"error": "Gemini API key not configured"}

        if not tagged_articles:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.5,
                "short_term_bias": "neutral",
                "long_term_bias": "neutral",
                "key_drivers": [],
                "summary": "No recent news available."
            }

        # Prepare prompt
        articles_text = "\n".join([
            f"- {art['title']} (Sentiment: {art['model_sentiment']}, Score: {art['model_score']:.2f})"
            for art in tagged_articles[:max_articles]
        ])

        prompt = f"""
        Analyze the following news articles for {symbol} and provide a structured trading signal summary.

        Articles:
        {articles_text}

        Based on these articles, provide a JSON response with:
        - overall_sentiment: "bullish", "bearish", or "neutral"
        - confidence: number between 0-1
        - short_term_bias: "bullish", "bearish", or "neutral"
        - long_term_bias: "bullish", "bearish", or "neutral"
        - key_drivers: array of main reasons (e.g., ["regulation", "earnings"])
        - summary: 1-2 sentence human-readable summary focused on trading implications

        Output only valid JSON.
        """

        try:
            self.logger.info(f"Sending {len(tagged_articles)} articles to Gemini for summarization")
            # Use the client to generate content with the correct model
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]

            parsed_result = json.loads(result_text.strip())
            # Validate the response structure
            if self.validate_gemini_response(parsed_result):
                self.logger.info("Gemini summarization completed successfully")
                return parsed_result
            else:
                self.logger.warning("Gemini response validation failed, using fallback")
                return {"error": "Invalid Gemini response structure"}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini JSON response: {str(e)}")
            return {"error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Gemini summarization failed: {str(e)}")
            return {"error": f"Gemini summarization failed: {str(e)}"}

    def validate_gemini_response(self, response):
        """
        Validate that Gemini response has required fields and correct types.
        """
        required_fields = {
            'overall_sentiment': str,
            'confidence': (int, float),
            'short_term_bias': str,
            'long_term_bias': str,
            'key_drivers': list,
            'summary': str
        }

        for field, expected_type in required_fields.items():
            if field not in response:
                self.logger.warning(f"Missing required field: {field}")
                return False
            if not isinstance(response[field], expected_type):
                self.logger.warning(f"Field {field} has wrong type: expected {expected_type}, got {type(response[field])}")
                return False

        # Validate sentiment values
        valid_sentiments = ['bullish', 'bearish', 'neutral']
        for field in ['overall_sentiment', 'short_term_bias', 'long_term_bias']:
            if response[field] not in valid_sentiments:
                self.logger.warning(f"Invalid sentiment value for {field}: {response[field]}")
                return False

        # Validate confidence range
        if not (0 <= response['confidence'] <= 1):
            self.logger.warning(f"Confidence out of range: {response['confidence']}")
            return False

        return True

    def fallback_sentiment_analysis(self, tagged_articles, symbol, max_articles=5):
        """
        Fallback sentiment analysis using aggregated FinBERT scores when Gemini fails.
        """
        if not tagged_articles:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.5,
                "short_term_bias": "neutral",
                "long_term_bias": "neutral",
                "key_drivers": [],
                "summary": "No recent news available (fallback analysis)."
            }

        # Aggregate sentiment from top articles
        top_articles = tagged_articles[:max_articles]

        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_score = 0
        key_drivers = []

        for art in top_articles:
            sent = art['model_sentiment']
            if sent in sentiment_counts:
                sentiment_counts[sent] += 1
            total_score += art['model_score']

            # Extract potential drivers from title
            title_lower = art['title'].lower()
            if 'regulation' in title_lower or 'sec' in title_lower:
                key_drivers.append('regulation')
            elif 'hack' in title_lower or 'breach' in title_lower:
                key_drivers.append('security')
            elif 'earnings' in title_lower or 'revenue' in title_lower:
                key_drivers.append('earnings')

        # Determine overall sentiment
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        avg_score = total_score / len(top_articles)

        # Map to trading terms
        if max_sentiment == "positive":
            overall = "bullish"
        elif max_sentiment == "negative":
            overall = "bearish"
        else:
            overall = "neutral"

        # Confidence based on consensus
        max_count = sentiment_counts[max_sentiment]
        confidence = min(max_count / len(top_articles), 0.9)  # Cap at 0.9 for fallback

        return {
            "overall_sentiment": overall,
            "confidence": round(confidence, 2),
            "short_term_bias": overall,
            "long_term_bias": "neutral",  # Fallback doesn't distinguish timeframes
            "key_drivers": list(set(key_drivers))[:3],  # Unique drivers, max 3
            "summary": f"News sentiment analysis shows {overall} bias with {confidence:.0%} confidence based on {len(top_articles)} articles (fallback analysis)."
        }

    def get_news_signal(self, symbol, days=10, max_articles=5):
        """
        Main method to get news-based trading signal for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'AAPL')
            days: Number of days to look back for news
            max_articles: Maximum number of articles to analyze (default: 5)
        """
        articles = self.fetch_news(symbol, days)
        if isinstance(articles, dict) and 'error' in articles:
            return {"ticker": symbol, "error": articles['error']}

        tagged_articles = self.analyze_sentiment(articles)

        # Try Gemini summarization first
        signal = self.summarize_with_gemini(tagged_articles, symbol, max_articles)

        # Fallback to aggregated sentiment if Gemini fails
        if isinstance(signal, dict) and 'error' in signal:
            signal = self.fallback_sentiment_analysis(tagged_articles, symbol, max_articles)

        # Add ticker and article count
        signal["ticker"] = symbol
        signal["article_count"] = len(tagged_articles)

        return signal

# Example usage
if __name__ == "__main__":
    agent = NewsAgent()
    result = agent.get_news_signal("BTC", days=10)
    print(json.dumps(result, indent=2))