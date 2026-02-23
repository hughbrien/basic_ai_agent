
#!/usr/bin/env python3
"""
ChatBox (LangChain 0.3.x+)

Providers: OpenAI, Anthropic, Groq, Ollama (Llama3), Grok (xAI via OpenAI-compatible endpoint)
Tools: Calculator (safe AST), optional TavilySearch (if TAVILY_API_KEY is set)
Persistence: SQLite chat history (per session_id) + SQLite response cache
Audit: JSONL per-request metadata (provider, model, prompts, latency, token usage when available, tool calls)
CLI: Sensible defaults; override via env or CLI flags

Requires: see requirements.txt — all langchain-* packages must be on the 0.3.x
branch together. Mixing 0.2.x langchain with 0.3.x langchain-core causes a
`ModuleNotFoundError: No module named 'langchain_core.pydantic_v1'` because
the pydantic v1 shim was removed in langchain-core 0.3.
"""

import os
import time
import json
import argparse
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

# -------- LangChain core (0.3.x+) --------
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub

# Persistent chat history (community in 0.2.x+)
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Providers (latest package names)
from langchain_openai import ChatOpenAI           # OpenAI + Grok (OpenAI-compatible)
from langchain_anthropic import ChatAnthropic     # Anthropic
from langchain_groq import ChatGroq               # Groq
from langchain_ollama import ChatOllama  # Ollama local server (supports bind_tools)

# Optional Tavily search tool (latest location in community pkg)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    HAS_TAVILY = True
except Exception:
    HAS_TAVILY = False


print ("Initializing Traceloop...")
from traceloop.sdk import Traceloop

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OTEL_URL = os.getenv("OTEL_URL")
OTEL_TOKEN = os.getenv("OTEL_TOKEN", "XYZ")


Traceloop.init(
    app_name="basic-ai-agent",
    api_endpoint="http://localhost:4318",
    headers={"Authorization": "Api-Token " + OTEL_TOKEN },
    disable_batch = False
)
print ("Finished Initializing Tracelook...")

# --------------------------
# Configuration & Providers
# --------------------------
@dataclass
class ProviderConfig:
    provider: str
    model: str
    temperature: float = 0.2
    timeout_s: int = 60
    streaming: bool = False


def make_llm(cfg: ProviderConfig):
    """Return a LangChain ChatModel for the selected provider."""
    p = cfg.provider.lower()

    if p == "openai":
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
            streaming=cfg.streaming,
        )

    elif p == "anthropic":
        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
            streaming=cfg.streaming,
        )

    elif p == "groq":
        return ChatGroq(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
            streaming=cfg.streaming,
        )

    elif p == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            base_url=base_url,
            model=cfg.model,
            temperature=cfg.temperature,
        )

    elif p == "grok":
        # xAI Grok via OpenAI-compatible endpoint
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is required for provider=grok")
        return ChatOpenAI(
            model=cfg.model,
            base_url=base_url,
            api_key=api_key,
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
            streaming=cfg.streaming,
        )

    else:
        raise ValueError(f"Unsupported provider: {cfg.provider}")


# --------------------------
# Tools
# --------------------------
def _calculator(expr: str) -> str:
    """Evaluate a safe arithmetic expression (supports +, -, *, /, **, (), floats)."""
    import ast
    import operator as op

    # Ollama and some other models may pass a JSON-schema dict instead of a
    # plain string (e.g. {'type': 'string'}).  Extract the nested 'expr' key
    # when that happens so the tool degrades gracefully rather than crashing.
    if isinstance(expr, dict):
        expr = expr.get("expr", "")
    if not isinstance(expr, str) or not expr.strip():
        return "Error: expected a math expression string, e.g. '2 + 2'"

    OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -eval_node(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.Expr):
            return eval_node(node.value)
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expr, mode="eval")
        return str(eval_node(tree.body))
    except Exception as e:
        return f"Error: {e}"


# Wrap in a plain Tool (not the @tool / StructuredTool decorator) so that
# LangChain passes the model's raw string directly to _calculator without
# going through Pydantic schema validation.  This avoids the
# "Input should be a valid string [input_value={'type': 'string'}]" crash
# that occurs when some LLMs (e.g. Ollama llama3) echo the JSON-schema
# definition back as the argument value instead of the actual expression.
calculator = Tool(
    name="calculator",
    func=_calculator,
    description=(
        "Evaluate a safe arithmetic expression. "
        "Input must be a math expression string, e.g. '2 + 2' or '(10 * 5) / 2'."
    ),
)


# --------------------------
# Weather tool (Open-Meteo — free, no API key required)
# --------------------------
# WMO weather interpretation codes → human-readable descriptions
_WMO_CODES: Dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


class WeatherInput(BaseModel):
    location: str = Field(
        description="City name with optional state/country, e.g. 'Bradenton FL' or 'Paris, France'"
    )


def _get_weather(location: str) -> str:
    """Return current weather conditions for *location* using Open-Meteo."""
    try:
        # 1. Geocode the location name → lat/lon
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"Could not find location: {location}"

        place = geo_data["results"][0]
        lat = place["latitude"]
        lon = place["longitude"]
        city = place.get("name", location)
        state = place.get("admin1", "")
        country = place.get("country", "")
        display = f"{city}, {state}" if state else city
        if country:
            display += f", {country}"

        # 2. Fetch current conditions
        wx_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_direction_10m",
                ],
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
            },
            timeout=10,
        )
        wx_resp.raise_for_status()
        current = wx_resp.json().get("current", {})

        wmo = current.get("weather_code", 0)
        condition = _WMO_CODES.get(wmo, f"Weather code {wmo}")

        return (
            f"Current weather for {display}:\n"
            f"  Condition:   {condition}\n"
            f"  Temperature: {current.get('temperature_2m', 'N/A')}°F"
            f" (feels like {current.get('apparent_temperature', 'N/A')}°F)\n"
            f"  Humidity:    {current.get('relative_humidity_2m', 'N/A')}%\n"
            f"  Wind:        {current.get('wind_speed_10m', 'N/A')} mph"
        )
    except Exception as exc:
        return f"Error fetching weather: {exc}"


weather_tool = StructuredTool.from_function(
    func=_get_weather,
    name="get_weather",
    description=(
        "Get current weather conditions for any city or location. "
        "Use this for any weather-related questions. "
        "Input: location name such as 'Bradenton FL', 'New York', or 'London UK'."
    ),
    args_schema=WeatherInput,
)


def build_tools() -> List[Tool]:
    tools: List[Tool] = [calculator, weather_tool]
    if HAS_TAVILY and os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=3))
    return tools


# --------------------------
# Prompt & Agent
# --------------------------
def build_agent(llm, tools: List[Tool]) -> AgentExecutor:
    """
    Tool-calling agent for LangChain 0.3.x+:
    Uses the model's native tool-calling API when supported; falls back to
    ReAct (text-based) agent for models that don't implement bind_tools
    (e.g. older Ollama models).
    """
    system_text = (
        "You are ChatBox, an enterprise-ready assistant.\n"
        "Be concise, correct, and use tools when helpful."
    )

    try:
        # Verify the model supports bind_tools before committing to the agent type
        llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # list[BaseMessage]
        ])

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    except NotImplementedError:
        # Fallback: ReAct agent for models without native tool-calling support
        react_prompt = hub.pull("hwchase17/react-chat")
        agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


# --------------------------
# Persistent History & Cache
# --------------------------
def get_history(session_id: str) -> SQLChatMessageHistory:
    conn = "sqlite:///chat_history.db"
    return SQLChatMessageHistory(connection=conn, session_id=session_id)


def init_response_cache(path: str = ".langchain_cache.db"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=path))


# --------------------------
# Audit Logging (metadata)
# --------------------------
class AuditLogger(BaseCallbackHandler):
    """Capture metadata for each LLM call and persist to JSONL, including tool invocations."""
    def __init__(self, provider: str, model: str, log_path: str = "logs/llm_requests.jsonl"):
        self.provider = provider
        self.model = model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._record: Dict[str, Any] = {}
        self._tools_used: List[Dict[str, Any]] = []

    # LLM lifecycle
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self._record = {
            "ts_start": time.time(),
            "provider": self.provider,
            "model": self.model,
            "prompts": prompts,
            "invocation_params": kwargs.get("invocation_params", {}),
        }

        self._tools_used = []

    def on_llm_end(self, response, **kwargs):
        self._record["ts_end"] = time.time()
        self._record["latency_s"] = self._record["ts_end"] - self._record["ts_start"]
        # Token usage and other metadata (when available)
        try:
            self._record["llm_output"] = response.llm_output
            # Convenient top-level token usage if present
            if isinstance(response.llm_output, dict) and "token_usage" in response.llm_output:
                self._record["token_usage"] = response.llm_output["token_usage"]
        except Exception:
            self._record["llm_output"] = None
        # Tools list
        if self._tools_used:
            self._record["tools_used"] = self._tools_used
        # Persist
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._record) + "\n")

    def on_llm_error(self, error: Exception, **kwargs):
        self._record["ts_end"] = time.time()
        self._record["error"] = str(error)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._record) + "\n")

    # Tool lifecycle
    def on_tool_start(self, tool, input_str: str, **kwargs):
        entry = {"tool": getattr(tool, "name", str(tool)), "input": input_str, "ts": time.time()}
        self._tools_used.append(entry)

    def on_tool_end(self, output: str, **kwargs):
        # Backfill the last tool call with output and duration if possible
        if self._tools_used:
            last = self._tools_used[-1]
            last["output"] = output
            last["ts_end"] = time.time()
            last["latency_s"] = last["ts_end"] - last["ts"]

    def on_tool_error(self, error: Exception, **kwargs):
        if self._tools_used:
            last = self._tools_used[-1]
            last["error"] = str(error)
            last["ts_end"] = time.time()


# --------------------------
# CLI & Main
# --------------------------
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "groq": "llama-3.1-70b-versatile",
    "ollama": "llama3:latest",
    "grok": "grok-2",
}


def parse_args():
    p = argparse.ArgumentParser(description="LangChain ChatBox")
    p.add_argument(
        "--provider",
        choices=["openai", "anthropic", "groq", "ollama", "grok"],
        default=os.getenv("CHATBOX_PROVIDER", "ollama"),
        help="LLM provider (default from CHATBOX_PROVIDER env or 'ollama')",
    )
    p.add_argument(
        "--model",
        default=os.getenv("CHATBOX_MODEL"),
        help="Model name (default: provider-specific; see DEFAULT_MODELS)",
    )
    p.add_argument("--session", default=os.getenv("CHATBOX_SESSION", "default"))
    p.add_argument("--temperature", type=float, default=float(os.getenv("CHATBOX_TEMPERATURE", 0.2)))
    p.add_argument("--timeout", type=int, default=int(os.getenv("CHATBOX_TIMEOUT", 60)))
    p.add_argument("--stream", action="store_true")
    return p.parse_args()


def check_env(provider: str):
    missing = []
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if provider == "grok" and not os.getenv("XAI_API_KEY"):
        missing.append("XAI_API_KEY")
    # Ollama uses local server; no key required
    if missing:
        raise RuntimeError(f"Missing environment variables for {provider}: {', '.join(missing)}")


def _extract_text(raw: Any) -> str:
    """Return plain text from agent output, stripping any JSON/list wrapper.

    Anthropic (and some other providers) return a list of content blocks
    such as ``[{'type': 'text', 'text': '...', 'index': 0}]`` instead of a
    plain string.  This helper handles Python lists/dicts directly as well
    as JSON-encoded strings.
    """
    # --- Native Python list of content blocks (e.g. Anthropic streaming) ---
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or item.get("output") or ""
                if t:
                    parts.append(str(t))
        return "\n".join(parts) if parts else str(raw)

    # --- Native Python dict ---
    if isinstance(raw, dict):
        for key in ("output", "content", "text", "answer", "response", "message"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        # Content block list inside a dict
        value = raw.get("content")
        if isinstance(value, list):
            parts = [
                b.get("text", "") for b in value
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined
        return str(raw)

    # --- String: return as-is unless it looks like JSON ---
    if not isinstance(raw, str):
        return str(raw)
    stripped = raw.strip()
    if not stripped.startswith(("{", "[")):
        return raw  # fast path: already plain text
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return raw  # not valid JSON – return as-is
    # Recurse now that we have a real Python object
    return _extract_text(parsed)


def main():
    args = parse_args()
    # Apply provider-specific default model if --model / CHATBOX_MODEL not set
    if not args.model:
        args.model = DEFAULT_MODELS.get(args.provider, "llama3:latest")
    check_env(args.provider)

    # Response cache (persists across runs)
    init_response_cache(".langchain_cache.db")

    cfg = ProviderConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        timeout_s=args.timeout,
        streaming=args.stream,
    )
    llm = make_llm(cfg)

    tools = build_tools()
    agent = build_agent(llm, tools)

    history = get_history(args.session)
    callbacks = [AuditLogger(provider=args.provider, model=args.model)]

    banner = f"ChatBox ready | Provider={args.provider} | Model={args.model} | Session={args.session}"
    print(banner)
    print("Type your message, or Ctrl+C to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Persist human message
            history.add_message(HumanMessage(content=user_input))

            payload = {
                "input": user_input,
                "chat_history": history.messages,  # MUST be list[BaseMessage]
                # agent_scratchpad is injected by the agent; do not set it here
            }

            start = time.time()
            result: Dict[str, Any] = agent.invoke(payload, config={"callbacks": callbacks})
            latency = time.time() - start

            # Agents return dict with "output"; unwrap any JSON wrapper
            raw = result.get("output", str(result))
            text = _extract_text(raw)

            # Persist AI message
            history.add_message(AIMessage(content=text))

            print(f"\nAssistant ({args.provider}/{args.model} | {latency:.2f}s):\n{text}\n")

    except KeyboardInterrupt:
        print("\nExiting. Bye!")


if __name__ == "__main__":
    main()



