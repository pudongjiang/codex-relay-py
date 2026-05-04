# codex-relay-py

Python reimplementation of [codex-relay](https://github.com/anthropics/codex-relay): a translation bridge between OpenAI's **Responses API** and the widely-supported **Chat Completions API**.

OpenAI Codex CLI speaks the Responses API, but many providers (DeepSeek, OpenRouter, etc.) only offer the Chat Completions API. `codex-relay-py` sits in the middle and translates between the two protocols, so Codex CLI can work with any Chat Completions-compatible provider.

## Features

- **Protocol translation** — Converts Responses API requests to Chat Completions, and back
- **Streaming support** — Full SSE stream translation (text deltas + tool call deltas)
- **Reasoning content** — Preserves `reasoning_content` across multi-turn tool call conversations
- **Session management** — Maintains conversation history via `previous_response_id`
- **Model name mapping** — Map Codex model names (e.g. `gpt-5.4-mini`) to provider models (e.g. `deepseek-v4-pro`)
- **LLM request/response logging** — Optional verbose logging of upstream requests and responses (`CODEX_RELAY_LOG_LLM=1`)

## Quick start

```bash
# Set your API key
export CODEX_RELAY_API_KEY="sk-your-key-here"

# Start the relay (defaults to DeepSeek on port 4446)
./start.sh
```

Or run directly:

```bash
PYTHONPATH=. .venv/bin/python -m py \
  --port 4446 \
  --upstream "https://api.deepseek.com/v1" \
  --api-key "$CODEX_RELAY_API_KEY"
```

## Configuration

| Argument | Env variable | Default | Description |
|---|---|---|---|
| `--port` | `CODEX_RELAY_PORT` | `4446` | Port to listen on |
| `--upstream` | `CODEX_RELAY_UPSTREAM` | `https://api.deepseek.com/v1` | Upstream provider base URL |
| `--api-key` | `CODEX_RELAY_API_KEY` | (empty) | API key for the upstream provider |
| `--model-map` | `CODEX_RELAY_MODEL_MAP` | `'gpt-5.4-mini=deepseek-v4-flash'` | Model name mappings (`from=to,from2=to2`) |

### LLM logging

Enable verbose logging of requests sent to and responses received from the upstream LLM:

```bash
export CODEX_RELAY_LOG_LLM=1  # enabled by default; set to 0 to disable
```

### Model mapping

Map multiple Codex model names to your provider's models:

```bash
export CODEX_RELAY_MODEL_MAP="gpt-5.4-mini=deepseek-v4-pro,gpt-5.4=deepseek-v4-flash"
```

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/responses` | Responses API → Chat Completions translation |
| `GET /v1/models` | Proxied models list from upstream |

## Programmatic usage

```python
from py import start

thread = start(
    port=4446,
    upstream="https://api.deepseek.com/v1",
    api_key="sk-your-key",
    model_map={"gpt-5.4-mini": "deepseek-v4-pro"},
)
# Server runs as a daemon thread; call thread.join() if you want to block
```

## Requirements

- Python 3.11+
- Dependencies: `fastapi`, `uvicorn`, `httpx`, `pydantic`

```bash
pip install fastapi uvicorn httpx pydantic
```

## Project structure

```
codex-relay-py/
├── start.sh            # Launcher script
├── py/
│   ├── __init__.py     # Package exports (start, validate_upstream)
│   ├── __main__.py     # Entry point for `python -m py`
│   ├── server.py       # FastAPI HTTP server + CLI
│   ├── stream.py       # SSE streaming translation
│   ├── translate.py    # Protocol translation (request/response)
│   ├── session.py      # In-memory session store
│   ├── types.py        # Pydantic/dataclass models
│   └── llm_log.py      # LLM request/response logging
└── README.md
```

## License

MIT
