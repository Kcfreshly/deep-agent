"""Canadian study and immigration advisory agent construction."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepagents import MemoryMiddleware, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parent
MEMORY_ROOT = PROJECT_ROOT / "memory"
PROMPT_ROOT = PROJECT_ROOT / "prompts"
DEFAULT_MODEL = "deepseek:deepseek-v4-flash"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_OPENAI_SEARCH_MODEL = "gpt-5.6-luna"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MEMORY_FILES = (
    "conversation.md",
    "profile.md",
    "education.md",
    "work_experience.md",
    "finances.md",
    "family.md",
    "immigration.md",
    "career_goals.md",
    "preferences.md",
    "applications.md",
    "evidence_log.md",
    "decisions.md",
    "goals.md",
)
MEMORY_SOURCES = [f"/memory/{filename}" for filename in MEMORY_FILES]
SPECIALIST_MEMORY: dict[str, tuple[str, ...]] = {
    "admission-search-specialist": (
        "conversation.md",
        "education.md",
        "work_experience.md",
        "finances.md",
        "career_goals.md",
        "preferences.md",
        "applications.md",
    ),
    "admission-strategy-specialist": (
        "conversation.md",
        "education.md",
        "work_experience.md",
        "career_goals.md",
        "applications.md",
        "evidence_log.md",
    ),
    "canadian-immigration-specialist": (
        "conversation.md",
        "profile.md",
        "education.md",
        "work_experience.md",
        "finances.md",
        "family.md",
        "immigration.md",
        "preferences.md",
    ),
    "labour-settlement-specialist": (
        "conversation.md",
        "profile.md",
        "work_experience.md",
        "family.md",
        "career_goals.md",
        "preferences.md",
    ),
}
OPENAI_WEB_SEARCH_TOOL: dict[str, object] = {
    "type": "web_search",
    "search_context_size": "medium",
}


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> list[dict[str, object]]:
    """Search the web and return concise source snippets with URLs."""
    if os.environ.get("OPENAI_API_KEY"):
        return openai_web_search(query=query, max_results=max_results)
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY or TAVILY_API_KEY is required to use web search."
        raise RuntimeError(msg)
    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "topic": topic,
            "search_depth": "basic",
            "max_results": max(1, min(max_results, 8)),
            "include_answer": False,
            "include_raw_content": False,
        }
    ).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 -- fixed HTTPS API
            body = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        msg = f"Web search failed: {error}"
        raise RuntimeError(msg) from error
    results = body.get("results", [])
    return [
        {
            key: result[key]
            for key in ("title", "url", "content", "score", "published_date")
            if key in result
        }
        for result in results
        if isinstance(result, dict)
    ]


def _response_text(response: dict[str, Any]) -> str:
    """Extract the assistant text from an OpenAI Responses API payload."""
    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        chunks.append(output_text)
    return "\n".join(chunks).strip()


def _response_sources(response: dict[str, Any]) -> list[dict[str, str]]:
    """Extract URL citations from web search calls and text annotations."""
    sources: dict[str, dict[str, str]] = {}
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if isinstance(action, dict):
            for source in action.get("sources", []):
                if isinstance(source, dict) and isinstance(source.get("url"), str):
                    url = source["url"]
                    sources[url] = {"url": url}
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []):
                if not isinstance(annotation, dict):
                    continue
                url = annotation.get("url")
                if not isinstance(url, str):
                    continue
                source = sources.setdefault(url, {"url": url})
                title = annotation.get("title")
                if isinstance(title, str):
                    source["title"] = title
    return list(sources.values())


def openai_web_search(query: str, max_results: int = 5) -> list[dict[str, object]]:
    """Use OpenAI only for hosted web search and summarize cited findings."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY is required for OpenAI-backed web search."
        raise RuntimeError(msg)
    search_context_size = os.environ.get("OPENAI_SEARCH_CONTEXT_SIZE", "low")
    payload = json.dumps(
        {
            "model": os.environ.get(
                "OPENAI_SEARCH_MODEL", DEFAULT_OPENAI_SEARCH_MODEL
            ),
            "input": (
                "Search the web and return a concise, source-grounded answer. "
                f"Include no more than {max(1, min(max_results, 8))} source URLs.\n\n"
                f"Query: {query}"
            ),
            "tools": [
                {
                    "type": "web_search",
                    "search_context_size": search_context_size,
                }
            ],
            "tool_choice": {"type": "web_search"},
        }
    ).encode("utf-8")
    request = Request(
        OPENAI_RESPONSES_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:  # noqa: S310 -- fixed HTTPS API
            body = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        msg = f"OpenAI web search failed: {error}"
        raise RuntimeError(msg) from error
    return [
        {
            "title": "OpenAI web search result",
            "url": source.get("url", ""),
            "content": _response_text(body),
            **({"source_title": source["title"]} if "title" in source else {}),
        }
        for source in _response_sources(body)
    ] or [
        {
            "title": "OpenAI web search result",
            "url": "",
            "content": _response_text(body),
        }
    ]


def resolve_advisory_model(model: str, *, max_tokens: int | None = None) -> BaseChatModel:
    """Resolve an advisory model, including DeepSeek's compatible endpoint."""
    if model.startswith("deepseek:"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            msg = "DEEPSEEK_API_KEY is required for a `deepseek:<model>` model."
            raise RuntimeError(msg)
        model_name = model.removeprefix("deepseek:")
        if not model_name:
            msg = "A DeepSeek model name is required after `deepseek:`."
            raise ValueError(msg)
        extra_body: dict[str, object] = {"thinking": {"type": "disabled"}}
        if max_tokens is not None:
            extra_body["max_tokens"] = max_tokens
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_API_BASE", DEEPSEEK_API_BASE),
            extra_body=extra_body,
            max_retries=2,
        )
    kwargs: dict[str, object] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(model, **kwargs)


def _research_tools(model: str) -> list[dict[str, object] | BaseTool]:
    """Select the search implementation supported by the model provider."""
    if model.startswith("openai:"):
        return [OPENAI_WEB_SEARCH_TOOL]
    if model.startswith("deepseek:"):
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("TAVILY_API_KEY")):
            return []
        return [internet_search]
    msg = "Supported model providers are `deepseek:` and `openai:`."
    raise ValueError(msg)


def _prompt(filename: str) -> str:
    """Load a required prompt file."""
    path = PROMPT_ROOT / filename
    if not path.is_file():
        msg = f"Required prompt file is missing: {path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def enable_langsmith_tracing() -> None:
    """Enable LangSmith tracing for every advisory-agent run."""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "canadian-study-advisor")


def _specialist(
    *,
    name: str,
    description: str,
    prompt_file: str,
    model: str | BaseChatModel,
    backend: CompositeBackend,
    tools: list[dict[str, object] | BaseTool],
) -> SubAgent:
    """Build an isolated specialist with current research and memory context."""
    sources = [f"/memory/{filename}" for filename in SPECIALIST_MEMORY[name]]
    system_prompt = f"{_prompt('evidence_policy.md')}\n\n{_prompt(prompt_file)}"
    return SubAgent(
        name=name,
        description=description,
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[MemoryMiddleware(backend=backend, sources=sources)],
    )


def create_advisory_agent(
    model: str | None = None,
) -> Runnable[dict[str, object], dict[str, object]]:
    """Create the Canadian study and immigration advisory agent."""
    load_dotenv(PROJECT_ROOT / ".env")
    enable_langsmith_tracing()
    selected_model = model or os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL)
    research_tools = _research_tools(selected_model)
    resolved_model: str | BaseChatModel = selected_model
    if selected_model.startswith("deepseek:"):
        resolved_model = resolve_advisory_model(selected_model)
    memory_backend = FilesystemBackend(root_dir=MEMORY_ROOT, virtual_mode=True)
    backend = CompositeBackend(
        default=StateBackend(), routes={"/memory/": memory_backend}
    )
    specialists = [
        _specialist(
            name="admission-search-specialist",
            description="Find and verify Canadian programs that fit the user's profile, constraints, and long-term goals.",
            prompt_file="admission_search.md",
            model=resolved_model,
            backend=backend,
            tools=research_tools,
        ),
        _specialist(
            name="admission-strategy-specialist",
            description="Assess admission readiness and build an evidence-based application strategy.",
            prompt_file="admission_strategy.md",
            model=resolved_model,
            backend=backend,
            tools=research_tools,
        ),
        _specialist(
            name="canadian-immigration-specialist",
            description="Verify study-permit, DLI, PGWP, spouse-work, and potential PR implications from official sources.",
            prompt_file="immigration.md",
            model=resolved_model,
            backend=backend,
            tools=research_tools,
        ),
        _specialist(
            name="labour-settlement-specialist",
            description="Compare Canadian labour markets, costs, family fit, and settlement tradeoffs by location.",
            prompt_file="labour_settlement.md",
            model=resolved_model,
            backend=backend,
            tools=research_tools,
        ),
    ]
    dated_prompt = (
        f"Current date: {date.today().isoformat()}\n\n{_prompt('coordinator.md')}"
    )
    return cast(
        "Runnable[dict[str, object], dict[str, object]]",
        create_deep_agent(
            model=resolved_model,
            tools=[],
            system_prompt=dated_prompt,
            subagents=specialists,
            backend=backend,
            memory=MEMORY_SOURCES,
            checkpointer=InMemorySaver(),
            name="canadian-study-immigration-advisor",
        ),
    )
