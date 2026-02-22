
#!/usr/bin/env python3
"""
ChatBox (LangChain 0.2.x+)

Providers: OpenAI, Anthropic, Groq, Ollama (Llama3), Grok (xAI via OpenAI-compatible endpoint)
Tools: Calculator (safe AST), optional TavilySearch (if TAVILY_API_KEY is set)
Persistence: SQLite chat history (per session_id) + SQLite response cache
Audit: JSONL per-request metadata (provider, model, prompts, latency, token usage when available, tool calls)
CLI: Sensible defaults; override via env or CLI flags
"""

import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

# -------- LangChain core (0.2.x+) --------
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

# Persistent chat history (community in 0.2.x+)
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Providers (latest package names)
from langchain_openai import ChatOpenAI           # OpenAI + Grok (OpenAI-compatible)
from langchain_anthropic import ChatAnthropic     # Anthropic
from langchain_groq import ChatGroq               # Groq
from langchain_community.chat_models import ChatOllama  # Ollama local server

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
    disable_batch=True
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
@tool("calculator", return_direct=False)
def calculator(expr: str) -> str:
    """Evaluate a safe arithmetic expression (supports +, -, *, /, **, (), floats)."""
    import ast
    import operator as op

    OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
    }

    def eval_node(node):
        if isinstance(node, ast.Num):  # Py<3.8
            return node.n
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


def build_tools() -> List[Tool]:
    tools: List[Tool] = [calculator]
    if HAS_TAVILY and os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=3))
    return tools


# --------------------------
# Prompt & Agent
# --------------------------
def build_agent(llm, tools: List[Tool]) -> AgentExecutor:
    """
    Tool-calling agent for LangChain 0.2.x+:
    Uses the model's native tool-calling API instead of text-based ReAct parsing,
    which is more reliable for modern models like Claude and GPT-4.
    """
    system_text = (
        "You are ChatBox, an enterprise-ready assistant.\n"
        "Be concise, correct, and use tools when helpful."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # list[BaseMessage]
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


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

            # Agents return dict with "output"
            text = result.get("output", str(result))

            # Persist AI message
            history.add_message(AIMessage(content=text))

            print(f"\nAssistant ({args.provider}/{args.model} | {latency:.2f}s):\n{text}\n")

    except KeyboardInterrupt:
        print("\nExiting. Bye!")


if __name__ == "__main__":
    main()



