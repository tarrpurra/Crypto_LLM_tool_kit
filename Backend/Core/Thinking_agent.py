import os
import json
import time
from typing import Any, Dict, Optional

from openai import OpenAI


def load_model_from_config():
    """Load model from api_keys.json config file."""
    config_path = os.path.join(os.path.dirname(__file__), '../configs/api_keys.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("Model", "mistralai/devstral-251")
    except (FileNotFoundError, json.JSONDecodeError):
        return "mistralai/devstral-251"

SYSTEM_INSTRUCTION = """ROLE: You are a smart fund manager and a very professional trader who can trade in various instruments like (ETF, Crypto, Stocks, futures etc).
You evaluate and suggest what the person should do: go short, go long, or hold.

You will get information from multiple agents:
- NEWS_Agent: top current news about the stock/crypto
- Sentiment_agent: sentiment + confidence score from news
- Technical_agent: technicals (MACD, RSI, candles, whales/transactions for crypto) + confidence score
- ML-Agent: This agent predicts future price (XGBOOST/LSTM) when provide confidence score
- Risk-agent: risk profile / risk capacity
- User-agent: portfolio + positions

Task:
Use your reasoning to weigh agents differently depending on context and output an actionable recommendation.
IMPORTANT: Output MUST be valid JSON only (no markdown, no extra text).
"""

JSON_OUTPUT_INSTRUCTION = """
Return ONLY valid JSON with this schema:
{
  "recommendation": "LONG|SHORT|HOLD",
  "confidence": 0-100,
  "key_drivers": [
    {"source_agent": "NEWS_Agent|Sentiment_agent|Technical_agent|ML-Agent|Risk-agent|User-agent", "summary": "...", "weight": 0-1}
  ],
  "risk_plan": {
    "position_size_pct": 0-100,
    "entry": {"type": "market|limit", "price": null_or_number},
    "stop_loss": {"type": "price|pct|atr", "value": number},
    "take_profit": {"type": "price|pct|rr", "value": number},
    "time_horizon": "intraday|swing|long_term",
    "invalidations": ["..."]
  },
  "notes": ["..."],
  "missing_inputs": ["..."],
  "Reasoning": "..."
}
Rules:
- If critical info is missing, still output JSON and fill missing_inputs.
- confidence must reflect data quality + agreement between agents.
"""

class VibeTraderThinker:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        # Load model from config if not provided
        if model is None:
            model = load_model_from_config()
        
        # FIX 1: Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not provided. Set OPENROUTER_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        # FIX 2: Ensure API key has proper format
        if not api_key.startswith("sk-"):
            print(f"⚠️  Warning: OpenRouter API keys usually start with 'sk-'. "
                  f"Your key starts with: {api_key[:10]}...")
        
        # FIX 3: Pass API key correctly with default_headers
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/vibetrader",  # Optional but recommended
                "X-Title": "VibeTrade Thinker",  # Optional but recommended
            }
        )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def run(self, user_prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION.strip()},
            {"role": "system", "content": JSON_OUTPUT_INSTRUCTION.strip()},
            {"role": "user", "content": user_prompt},
        ]

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    # FIX 4: Enable JSON mode for better structured output
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )

                content = completion.choices[0].message.content or ""
                content = content.strip()

                # Some models occasionally wrap JSON in extra text. Try to extract JSON safely.
                data = self._safe_json_parse(content)

                # Optional: attach usage if present
                usage = getattr(completion, "usage", None)
                if usage:
                    data["_usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                return data

            except Exception as e:
                last_err = e
                print(f"⚠️  Attempt {attempt}/{self.max_retries} failed: {str(e)}")
                # exponential backoff
                if attempt < self.max_retries:
                    sleep_time = min(2 ** attempt, 8)
                    print(f"   Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        # First try direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # Hard fallback: return raw text so your pipeline doesn't break
        return {
            "recommendation": "HOLD",
            "confidence": 0,
            "key_drivers": [],
            "risk_plan": {
                "position_size_pct": 0,
                "entry": {"type": "market", "price": None},
                "stop_loss": {"type": "pct", "value": 0},
                "take_profit": {"type": "rr", "value": 0},
                "time_horizon": "swing",
                "invalidations": [],
            },
            "notes": ["Model did not return valid JSON. Raw output attached."],
            "missing_inputs": [],
            "Reasoning": "",
            "_raw": text,
        }


# Example usage
if __name__ == "__main__":
    # Make sure to set your API key:
    # export OPENROUTER_API_KEY=sk-or-v1-...
    
    try:
        thinker = VibeTraderThinker()
        
        # Example prompt
        result = thinker.run("""
        Analyze BTC based on these inputs:
        
        NEWS_Agent: Bitcoin surges past $100K on institutional adoption news
        Sentiment_agent: Bullish sentiment (85% confidence)
        Technical_agent: RSI at 72 (overbought), MACD bullish crossover
        ML-Agent: Predicts $105K in 7 days (70% confidence)
        Risk-agent: User has medium risk tolerance
        User-agent: Currently 30% portfolio in BTC
        """)
        
        print(json.dumps(result, indent=2))
        
    except ValueError as e:
        print(f"❌ {e}")
        print("\nTo fix:")
        print("1. Get your API key from https://openrouter.ai/keys")
        print("2. Set it: export OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE")