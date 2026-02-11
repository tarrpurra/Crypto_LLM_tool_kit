import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from services.tool_registry import ToolRegistry


def load_api_config():
    """Load API configuration from api_keys.json file."""
    config_path = os.path.join(os.path.dirname(__file__), '../configs/api_keys.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


SYSTEM_INSTRUCTION = """
You are an agentic crypto trading analyst. You may call tools to gather data.
Rules:
- Use tools for any real-time or external data. Do NOT fabricate market data.
- You can call multiple tools sequentially and combine their outputs.
- Always use tool outputs as your source of truth.
- If critical data is missing, note it explicitly in the final response.
- Return ONLY valid JSON as the final response (no markdown).
""".strip()

FINAL_SCHEMA_INSTRUCTION = """
Return JSON with this schema:
{
  "summary": "...",
  "tools_used": ["tool_name", "..."],
  "analysis": {
    "market_state": "...",
    "signals": ["..."],
    "supporting_data": {"...": "..."}
  },
  "risk": {
    "volatility": "...",
    "risk_reward": "...",
    "notes": ["..."]
  },
  "action": {
    "bias": "LONG|SHORT|HOLD",
    "confidence": 0-100,
    "time_horizon": "intraday|swing|long_term",
    "conditions": ["..."]
  },
  "data_gaps": ["..."]
}
""".strip()


class LLMToolOrchestrator:
    """Run an LLM-driven tool orchestration loop."""

    def __init__(
        self,
        api_key: Optional[str],
        tool_registry: ToolRegistry,
        tools: List[Dict[str, Any]],
        tool_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        max_steps: int = 6,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> None:
        config = load_api_config()
        if provider is None:
            provider = config.get("LLM_PROVIDER", "openrouter")
        provider = str(provider).strip().lower()

        if model is None:
            model = config.get("Model", "mistralai/devstral-251")

        if api_key is None:
            if provider == "together":
                api_key = config.get("TOGETHER_API_KEY") or os.getenv("TOGETHER_API_KEY")
                print(f"Using Together provider with API key: {api_key}")
            else:
                api_key = config.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

        if base_url is None:
            if provider == "together":
                base_url = "https://api.together.xyz/v1"
            elif provider == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
            else:
                base_url = config.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

        self.provider = provider
        if provider == "together":
            try:
                from together import Together
            except ImportError as exc:
                raise RuntimeError("Together provider selected but the 'together' package is not installed.") from exc
            if not api_key:
                raise RuntimeError("Together provider selected but no TOGETHER_API_KEY was found in api_keys.json.")
            self.client = Together(api_key=api_key)
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            if provider == "openrouter":
                self.client.headers = {"HTTP-Referer": "https://localhost:3000", "X-Title": "Trading Agent"}
        self.tool_registry = tool_registry
        self.tools = tools
        self.tool_handlers = tool_handlers
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.timeout = timeout

        self.logger = logging.getLogger("LLMToolOrchestrator")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def run(
        self,
        user_prompt: str,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the reasoning loop until the model returns a final response."""
        import traceback
        try:
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "system", "content": FINAL_SCHEMA_INSTRUCTION},
                {"role": "user", "content": user_prompt},
            ]

            tools_used: List[str] = []

            for step in range(1, self.max_steps + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    timeout=self.timeout,
                )

                message = response.choices[0].message
                tool_calls = message.tool_calls or []

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content or "",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in tool_calls
                            ],
                        }
                    )

                    for tool_call in tool_calls:
                        name = tool_call.function.name
                        tools_used.append(name)
                        args = self._safe_json_loads(tool_call.function.arguments or "{}")
                        log_line = f"🔧 Tool call ({step}): {name} args={args}"
                        self._log(log_line, log_callback)

                        handler = self.tool_handlers.get(name)
                        if not handler:
                            result = {"error": f"Tool '{name}' is not available."}
                            self.tool_registry.log_tool_invocation(name, False, 0.0, "Tool not available", "orchestrator")
                        else:
                            result = self._execute_tool(handler, name, args, log_callback)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": name,
                                "content": json.dumps(result),
                            }
                        )
                    continue

                final_content = message.content or ""
                if not final_content:
                    return {
                        "summary": "No response content returned.",
                        "tools_used": tools_used,
                        "analysis": {"market_state": "", "signals": [], "supporting_data": {}},
                        "risk": {"volatility": "", "risk_reward": "", "notes": []},
                        "action": {"bias": "HOLD", "confidence": 0, "time_horizon": "swing", "conditions": []},
                        "data_gaps": ["Model returned empty content."],
                    }

                return {"raw": final_content, "tools_used": tools_used}

            return {
                "summary": "Tool orchestration exceeded step limit.",
                "tools_used": tools_used,
                "analysis": {"market_state": "", "signals": [], "supporting_data": {}},
                "risk": {"volatility": "", "risk_reward": "", "notes": []},
                "action": {"bias": "HOLD", "confidence": 0, "time_horizon": "swing", "conditions": []},
                "data_gaps": ["Step limit reached before final response."],
            }
        except Exception as e:
            self.logger.error(f"Error in LLMToolOrchestrator.run: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return {
                "summary": f"Error during analysis: {e}",
                "tools_used": [],
                "analysis": {"market_state": "", "signals": [], "supporting_data": {}},
                "risk": {"volatility": "", "risk_reward": "", "notes": []},
                "action": {"bias": "HOLD", "confidence": 0, "time_horizon": "swing", "conditions": []},
                "data_gaps": [str(e)],
            }

    def _execute_tool(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        name: str,
        args: Dict[str, Any],
        log_callback: Optional[Callable[[str], None]],
    ) -> Dict[str, Any]:
        try:
            result = handler(args)
            self.tool_registry.log_tool_invocation(name, True, 0.0, "", "orchestrator")
            log_line = f"✅ Tool result ({name}): {self._summarize_result(result)}"
            self._log(log_line, log_callback)
            return result
        except Exception as exc:
            error_line = f"❌ Tool error ({name}): {exc}"
            self._log(error_line, log_callback)
            self.tool_registry.log_tool_invocation(name, False, 0.0, str(exc), "orchestrator")
            return {"error": str(exc)}

    @staticmethod
    def _safe_json_loads(payload: str) -> Dict[str, Any]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _summarize_result(result: Dict[str, Any]) -> str:
        if "error" in result:
            return result["error"]
        return "success"

    def _log(self, message: str, callback: Optional[Callable[[str], None]]) -> None:
        self.logger.info(message)
        if callback:
            callback(message)
