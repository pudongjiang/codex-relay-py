"""HTTP server: FastAPI app with CLI and programmatic entry points."""

import argparse
import logging
import os
import threading
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import llm_log
from .session import SessionStore
from .stream import translate_stream
from .translate import from_chat_response, to_chat_request
from .types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ResponsesRequest,
)

logger = logging.getLogger("codex_relay")


# ── Model name mapping ─────────────────────────────────────────────────────

def parse_model_map(raw: str) -> dict[str, str]:
    """Parse a ``key=val,key2=val2`` string into a model name mapping dict.

    Example: ``"gpt-5.4-mini=deepseek-v4-pro,gpt-5.4=deepseek-v4-flash"``
    """
    if not raw.strip():
        return {}
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"invalid model-map entry '{pair}': expected format 'from=to'"
            )
        k, v = pair.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


# ── URL validation ─────────────────────────────────────────────────────────

def validate_upstream(raw: str) -> str:
    """Validate that upstream URL is http/https with a host (SSRF protection)."""
    url = urlparse(raw.rstrip("/"))
    if url.scheme not in ("http", "https"):
        raise ValueError(
            f"upstream URL scheme must be http or https, got: {url.scheme}"
        )
    if not url.hostname:
        raise ValueError("upstream URL must have a host")
    return raw.rstrip("/")


def join_base(url: str) -> str:
    """Ensure URL ends with / for path joining."""
    return url if url.endswith("/") else f"{url}/"


# ── App factory ────────────────────────────────────────────────────────────

def create_app(
    upstream: str,
    api_key: str = "",
    model_map: dict[str, str] | None = None,
) -> FastAPI:
    """Build the FastAPI application with all routes."""

    if model_map is None:
        model_map = {}

    app = FastAPI(title="codex-relay-py")
    sessions = SessionStore()
    client: httpx.AsyncClient | None = None

    async def get_client() -> httpx.AsyncClient:
        nonlocal client
        if client is None:
            client = httpx.AsyncClient()
        return client

    @app.on_event("shutdown")
    async def shutdown():
        nonlocal client
        if client:
            await client.aclose()
            client = None

    # ── POST /v1/responses ────────────────────────────────────────────────

    @app.post("/v1/responses")
    async def handle_responses(request: Request):
        try:
            body = await request.body()
            req = ResponsesRequest.model_validate_json(body)
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return JSONResponse(
                status_code=422,
                content={"error": str(e)},
            )

        # Get prior history
        history: list[ChatMessage] = []
        if req.previous_response_id:
            history = sessions.get_history(req.previous_response_id)

        model = model_map.get(req.model, req.model)
        if model != req.model:
            logger.info(f"model mapped: {req.model} -> {model}")
        chat_req = to_chat_request(req, history, sessions, model_override=model)
        url = f"{join_base(upstream)}chat/completions"

        http_client = await get_client()

        if req.stream:
            response_id = sessions.new_id()
            chat_req.stream = True
            request_messages = list(chat_req.messages)

            stream_gen = translate_stream(
                client=http_client,
                url=url,
                api_key=api_key,
                chat_req=chat_req,
                response_id=response_id,
                sessions=sessions,
                prior_messages=history,
                request_messages=request_messages,
                model=model,
            )

            return StreamingResponse(
                stream_gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            chat_req.stream = False
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            chat_body = chat_req.to_dict()
            llm_log.log_request(model, url, chat_body)

            try:
                resp = await http_client.post(
                    url,
                    json=chat_body,
                    headers=headers,
                )
            except httpx.RequestError as e:
                logger.error(f"upstream error: {e}")
                return JSONResponse(
                    status_code=502,
                    content={"error": str(e)},
                )

            if resp.status_code >= 400:
                llm_log.log_response(model, resp.status_code, resp.text)
                logger.error(
                    f"upstream {resp.status_code}: {resp.text[:500]}"
                )
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": resp.text},
                )

            try:
                chat_obj = resp.json()
                llm_log.log_response(model, resp.status_code, resp.text)
            except Exception as e:
                logger.error(f"parse error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )

            choices = []
            for c in chat_obj.get("choices", []):
                from .types import ChatChoice
                msg_obj = c.get("message", {})
                msg = ChatMessage(
                    role=msg_obj.get("role", "assistant"),
                    content=msg_obj.get("content"),
                    reasoning_content=msg_obj.get("reasoning_content"),
                    tool_calls=msg_obj.get("tool_calls"),
                    tool_call_id=msg_obj.get("tool_call_id"),
                    name=msg_obj.get("name"),
                )
                choices.append(ChatChoice(message=msg))

            usage = None
            if "usage" in chat_obj:
                u = chat_obj["usage"]
                usage = ChatUsage(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    total_tokens=u.get("total_tokens", 0),
                )

            chat_resp = ChatResponse(choices=choices, usage=usage)

            assistant_msg = (
                choices[0].message
                if choices
                else ChatMessage(role="assistant", content="")
            )

            full_history = list(chat_req.messages)
            full_history.append(assistant_msg)
            response_id = sessions.save(full_history)

            (resp_body, _) = from_chat_response(response_id, model, chat_resp)
            return JSONResponse(content=resp_body.to_dict())

    # ── GET /v1/models ────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def handle_models():
        http_client = await get_client()
        url = f"{join_base(upstream)}models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = await http_client.get(url, headers=headers)
            if resp.status_code >= 400:
                logger.warning(f"upstream models: status {resp.status_code}")
                return JSONResponse(
                    content={"object": "list", "data": []}
                )
            return JSONResponse(content=resp.json())
        except httpx.RequestError as e:
            logger.warning(f"upstream models: request error: {e}")
            return JSONResponse(content={"object": "list", "data": []})
        except Exception as e:
            logger.warning(f"upstream models: parse error: {e}")
            return JSONResponse(content={"object": "list", "data": []})

    # ── Fallback ──────────────────────────────────────────────────────────

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def handle_fallback(request: Request, path: str):
        logger.warning(f"unhandled {request.method} {request.url.path}")
        return Response(content="not found", status_code=404)

    return app


# ── CLI ───────────────────────────────────────────────────────────────────

def start(
    port: int = 4444,
    upstream: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
    model_map: dict[str, str] | None = None,
) -> threading.Thread:
    """Start codex-relay as a background thread and return the thread handle."""

    upstream = validate_upstream(upstream)
    app = create_app(upstream=upstream, api_key=api_key, model_map=model_map)

    thread = threading.Thread(
        target=lambda: uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info",
        ),
        daemon=True,
    )
    thread.start()
    return thread


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="codex-relay-py",
        description="Responses API <-> Chat Completions bridge (Python)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CODEX_RELAY_PORT", "4446")),
        help="Port to listen on (env: CODEX_RELAY_PORT, default: 4446)",
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get(
            "CODEX_RELAY_UPSTREAM", "https://api.deepseek.com/v1"
        ),
        help="Upstream provider base URL (env: CODEX_RELAY_UPSTREAM)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CODEX_RELAY_API_KEY", ""),
        help="API key for upstream (env: CODEX_RELAY_API_KEY)",
    )
    parser.add_argument(
        "--model-map",
        default=os.environ.get("CODEX_RELAY_MODEL_MAP", "gpt-5.4-mini=deepseek-v4-flash"),
        help=(
            "Model name mapping: from=to,from2=to2 "
            "(env: CODEX_RELAY_MODEL_MAP). "
            "Example: 'gpt-5.4-mini=deepseek-v4-pro'"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    upstream = validate_upstream(args.upstream)
    model_map = parse_model_map(args.model_map)
    if model_map:
        logger.info(f"model map: {model_map}")

    app = create_app(upstream=upstream, api_key=args.api_key, model_map=model_map)

    logger.info(
        f"codex-relay-py listening on 127.0.0.1:{args.port} -> {upstream}"
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
