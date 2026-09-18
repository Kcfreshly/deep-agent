# Canadian Study & Immigration Advisory Agent

This local Deep Agents application coordinates four specialists:

- Canadian program search
- admission strategy
- Canadian immigration implications
- labour market and settlement

It starts in interview mode and keeps consented semantic memory in `memory/`. With an OpenAI key, DeepSeek specialists can call OpenAI hosted web search only when they need current sources. Immigration information is educational and is not legal advice.

Conversation persistence uses three Markdown files and requires no database:

- `memory/conversation.md` contains only the compact cross-thread summary injected into agent context.
- `memory/current_thread.md` contains the active thread transcript. The LangGraph checkpointer supplies this context to the running agent, so this file is not injected again.
- `memory/conversation_archive.md` is the append-only full chat archive and is never automatically sent to the model.

When a new CLI or notebook session starts, only the prior compact summary and the previous current-thread transcript are summarized. The growing archive is not reread, preventing its token cost from increasing forever.

`memory/goals.md` keeps the coordinator's current milestone, blocker, next action, and completion evidence across sessions. LangSmith tracing is enabled by default under the `canadian-study-advisor` project; set `LANGSMITH_API_KEY` in `.env` to send traces.

## Run from the terminal

```powershell
uv sync
uv run agent.py
```

You can also provide the first message directly:

```powershell
uv run agent.py "Start interview mode"
```

The default is `deepseek:deepseek-v4-flash`. Set `DEEP_AGENT_MODEL` in `.env` or pass `--model provider:model` to select another supported model. DeepSeek requires `DEEPSEEK_API_KEY`. If `OPENAI_API_KEY` is set, OpenAI is used for hosted web search when requested and for extracting relevant text from PDFs read by the agent; ordinary advisory turns still use DeepSeek. The search model defaults to `gpt-5.6-luna` and can be changed with `OPENAI_SEARCH_MODEL`; PDF extraction defaults to `gpt-5.6-luna` and can be changed with `OPENAI_PDF_MODEL`. `TAVILY_API_KEY` remains an optional fallback search provider.

## Run in Jupyter

Open `Untitled.ipynb`, select the `Python 3 (my-deep-agent)` kernel, restart the kernel, and run the cells in order.

## Data and privacy

- Markdown memory remains under `memory/` and is ignored by Git.
- The agent's filesystem backend cannot access `.env`; it is routed only to `memory/`.
- With the default configuration, advisory prompts are sent to DeepSeek. OpenAI is contacted only when `OPENAI_API_KEY` is configured and the agent searches the web or reads a PDF. For PDF reads, only the PDF and the current question are sent to OpenAI; extracted text is passed to DeepSeek, not the PDF block. Without an OpenAI key, the middleware replaces the PDF block with an explanation instead of sending an unsupported attachment to DeepSeek. Search queries go to Tavily only if `OPENAI_API_KEY` is absent and `TAVILY_API_KEY` is configured.
- LangSmith tracing is disabled by default because the profile can contain sensitive personal information. Set `ADVISOR_ENABLE_TRACING=true` only if the user understands and accepts tracing.
- Do not provide or store passport/UCI/account numbers, credentials, raw bank records, or raw medical/criminal records.
