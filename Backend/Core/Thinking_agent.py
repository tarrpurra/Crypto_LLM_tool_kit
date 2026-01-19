import os
import json
import time
from typing import Any, Dict, Optional

from openai import OpenAI

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
        api_key: str,
        model: str = "openrouter/mistralai/mixtral-8x7b-instruct",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
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
                    # If the endpoint supports it, uncomment:
                    # response_format={"type": "json_object"},
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
                # exponential backoff
                time.sleep(min(2 ** attempt, 8))

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
            "_raw": text,
        }


