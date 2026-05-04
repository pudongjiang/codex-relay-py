"""SSE streaming: translate Chat Completions SSE stream to Responses API SSE."""

import json
import logging
import uuid
from typing import AsyncIterator

import httpx

from . import llm_log
from .session import SessionStore
from .types import (
    ChatMessage,
    ChatRequest,
    parse_chat_stream_chunk,
    ChatStreamChunk,
)

logger = logging.getLogger(__name__)


class ToolCallAccum:
    """Accumulator for streaming tool call deltas."""

    def __init__(self):
        self.id: str = ""
        self.name: str = ""
        self.arguments: str = ""


def sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def translate_stream(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    chat_req: ChatRequest,
    response_id: str,
    sessions: SessionStore,
    prior_messages: list[ChatMessage],
    request_messages: list[ChatMessage],
    model: str,
) -> AsyncIterator[str]:
    """Translate an upstream Chat Completions SSE stream into Responses API SSE.

    Text response event sequence:
      response.created -> response.output_item.added -> response.output_text.delta*
      -> response.output_item.done -> response.completed

    Tool call response:
      response.created -> [accumulate deltas] -> response.output_item.added (function_call)
      -> response.function_call_arguments.delta -> response.output_item.done -> response.completed
    """
    msg_item_id = f"msg_{uuid.uuid4().hex}"

    # Emit response.created
    yield sse_event(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "status": "in_progress",
                "model": model,
            },
        },
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        stream_body = chat_req.to_dict()
        llm_log.log_stream_request(model, url, stream_body)

        async with client.stream(
            "POST",
            url,
            json=stream_body,
            headers=headers,
            timeout=None,
        ) as response:
            if response.status_code >= 400:
                body = ""
                try:
                    body = await response.aread()
                    body = body.decode()
                except Exception:
                    pass
                llm_log.log_stream_response(model, response.status_code, body)
                logger.error(f"upstream {response.status_code}: {body}")
                yield sse_event(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": response_id,
                            "status": "failed",
                            "error": {
                                "code": str(response.status_code),
                                "message": body,
                            },
                        },
                    },
                )
                return

            accumulated_text = ""
            accumulated_reasoning = ""
            tool_calls: dict[int, ToolCallAccum] = {}
            emitted_message_item = False

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # strip "data: "
                if data_str.strip() == "[DONE]":
                    break
                if not data_str.strip():
                    continue

                chunk = parse_chat_stream_chunk(data_str)
                if chunk is None:
                    logger.warning(f"chunk parse error — data: {data_str[:200]}")
                    continue

                for choice in chunk.choices:
                    # Reasoning/thinking content
                    rc = choice.delta.reasoning_content
                    if rc:
                        accumulated_reasoning += rc

                    # Text content
                    content = choice.delta.content or ""
                    if content:
                        if not emitted_message_item:
                            yield sse_event(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": 0,
                                    "item": {
                                        "type": "message",
                                        "id": msg_item_id,
                                        "role": "assistant",
                                        "content": [],
                                        "status": "in_progress",
                                    },
                                },
                            )
                            emitted_message_item = True

                        accumulated_text += content
                        yield sse_event(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "item_id": msg_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": content,
                            },
                        )

                    # Tool call deltas — accumulate by index
                    if choice.delta.tool_calls:
                        for dc in choice.delta.tool_calls:
                            entry = tool_calls.setdefault(
                                dc.index, ToolCallAccum()
                            )
                            if dc.id:
                                entry.id = dc.id
                            if dc.function:
                                if dc.function.name:
                                    entry.name += dc.function.name
                                if dc.function.arguments:
                                    entry.arguments += dc.function.arguments

            # Close message item if one was opened
            if emitted_message_item:
                yield sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": msg_item_id,
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": accumulated_text,
                                }
                            ],
                        },
                    },
                )

            # Emit function_call items for each accumulated tool call
            base_index = 1 if emitted_message_item else 0
            fc_items: list[dict] = []

            for rel_idx, tc in enumerate(
                sorted(tool_calls.values(), key=lambda tc: tc.id or "")
            ):
                fc_item_id = f"fc_{uuid.uuid4().hex}"
                output_index = base_index + rel_idx

                yield sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": fc_item_id,
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": "",
                            "status": "in_progress",
                        },
                    },
                )

                if tc.arguments:
                    yield sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": fc_item_id,
                            "output_index": output_index,
                            "delta": tc.arguments,
                        },
                    )

                yield sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": fc_item_id,
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "status": "completed",
                        },
                    },
                )

                fc_items.append({
                    "type": "function_call",
                    "id": fc_item_id,
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "status": "completed",
                })

            # Persist turn to session store
            for tc in tool_calls.values():
                if tc.id:
                    sessions.store_reasoning(
                        tc.id, accumulated_reasoning
                    )

            assistant_tool_calls = None
            if tool_calls:
                assistant_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in sorted(
                        tool_calls.values(), key=lambda tc: tc.id or ""
                    )
                ]

            assistant_msg = ChatMessage(
                role="assistant",
                content=accumulated_text or None,
                reasoning_content=accumulated_reasoning or None,
                tool_calls=assistant_tool_calls,
            )

            # Index reasoning by turn fingerprint
            if accumulated_reasoning:
                sessions.store_turn_reasoning(
                    request_messages, assistant_msg, accumulated_reasoning
                )

            messages = list(prior_messages)
            messages.append(assistant_msg)
            sessions.save_with_id(response_id, messages)

            # Build output array for response.completed
            output_items: list[dict] = []
            if emitted_message_item:
                output_items.append({
                    "type": "message",
                    "id": msg_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": accumulated_text}
                    ],
                })
            output_items.extend(fc_items)

            llm_log.log_stream_response(
                model,
                response.status_code,
                json.dumps(
                    {
                        "text": accumulated_text,
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in sorted(
                                tool_calls.values(), key=lambda tc: tc.id or ""
                            )
                        ],
                    },
                    ensure_ascii=False,
                ),
            )

            yield sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "model": model,
                        "output": output_items,
                    },
                },
            )

    except httpx.RequestError as e:
        llm_log.log_stream_response(model, 0, str(e))
        logger.error(f"upstream request failed: {e}")
        yield sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "status": "failed",
                    "error": {
                        "code": "connection_error",
                        "message": str(e),
                    },
                },
            },
        )
