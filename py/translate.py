"""Protocol translation: Responses API <-> Chat Completions API."""

from typing import Any, Optional

from .session import SessionStore
from .types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ContentPart,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)


def value_to_text(v: Any) -> str:
    """Collapse a Responses API content value (string or parts array) to plain text."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        # Array of content parts like [{"type": "input_text", "text": "hello"}]
        return "".join(
            part.get("text", "") if isinstance(part, dict) else ""
            for part in v
        )
    return str(v)


def convert_tool(tool: dict) -> dict:
    """Convert Responses API flat tool format to Chat Completions nested format.

    Responses API (flat):
      {"type":"function","name":"foo","description":"...","parameters":{...},"strict":false}

    Chat Completions (nested):
      {"type":"function","function":{"name":"foo","description":"...","parameters":{...}}}
    """
    # Already in Chat Completions format if it has a "function" sub-object
    if "function" in tool:
        return tool

    if tool.get("type") == "function":
        func: dict = {}
        for key in ("name", "description", "parameters", "strict"):
            if key in tool:
                func[key] = tool[key]
        return {"type": "function", "function": func}

    return tool


def to_chat_request(
    req: ResponsesRequest,
    history: list[ChatMessage],
    sessions: SessionStore,
) -> ChatRequest:
    """Convert a Responses API request + prior history into a Chat Completions request."""
    messages = list(history)

    # Prefer `instructions` (Codex CLI) over `system` (other clients)
    system_text = req.instructions or req.system
    if system_text:
        if not messages or messages[0].role != "system":
            messages.insert(
                0,
                ChatMessage(role="system", content=system_text),
            )

    # Append new input, mapping Responses API roles to Chat Completions roles
    if isinstance(req.input, str):
        messages.append(ChatMessage(role="user", content=req.input))
    else:
        items = req.input
        i = 0
        while i < len(items):
            item = items[i]
            item_type = item.get("type", "")

            if item_type == "function_call":
                # Collect consecutive function_call items into one assistant message
                grouped: list[dict] = []
                reasoning_content: Optional[str] = None

                while i < len(items):
                    cur = items[i]
                    if cur.get("type", "") != "function_call":
                        break
                    call_id = cur.get("call_id", "")
                    name = cur.get("name", "")
                    args = cur.get("arguments", "{}")
                    if reasoning_content is None:
                        reasoning_content = sessions.get_reasoning(call_id)
                    grouped.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                    i += 1

                msg = ChatMessage(
                    role="assistant",
                    content=None,
                    reasoning_content=reasoning_content,
                    tool_calls=grouped,
                )
                # Fallback: try turn-level fingerprint
                if msg.reasoning_content is None:
                    msg.reasoning_content = sessions.get_turn_reasoning(
                        messages, msg
                    )
                messages.append(msg)
            else:
                if item_type == "function_call_output":
                    call_id = item.get("call_id", "")
                    output = item.get("output", "")
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=str(output),
                            tool_call_id=call_id,
                        )
                    )
                else:
                    # Regular user/assistant/developer message
                    role = item.get("role", "user")
                    if role == "developer":
                        role = "system"
                    content = value_to_text(item.get("content"))
                    msg = ChatMessage(role=role, content=content)
                    # For assistant messages, try to recover reasoning_content
                    if msg.role == "assistant":
                        msg.reasoning_content = sessions.get_turn_reasoning(
                            messages, msg
                        )
                    messages.append(msg)
                i += 1

    # Keep only function-type tools; providers don't accept OpenAI built-ins
    tools = [
        convert_tool(t)
        for t in req.tools
        if t.get("type") == "function"
    ]

    return ChatRequest(
        model=req.model,
        messages=messages,
        tools=tools,
        temperature=req.temperature,
        max_tokens=req.max_output_tokens,
        stream=req.stream,
    )


def from_chat_response(
    id: str,
    model: str,
    chat: ChatResponse,
) -> tuple[ResponsesResponse, list[ChatMessage]]:
    """Convert a Chat Completions response into a Responses API response."""
    if chat.choices:
        choice_msg = chat.choices[0].message
    else:
        choice_msg = ChatMessage(role="assistant", content="")

    text = choice_msg.content or ""
    usage = chat.usage or ChatUsage()

    response = ResponsesResponse(
        id=id,
        object="response",
        model=model,
        output=[
            ResponsesOutputItem(
                type="message",
                role="assistant",
                content=[
                    ContentPart(type="output_text", text=text),
                ],
            )
        ],
        usage=ResponsesUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        ),
    )

    return response, [choice_msg]
