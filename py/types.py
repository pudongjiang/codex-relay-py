import json
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ── Responses API (inbound from Codex CLI) ──────────────────────────────────

class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, list[dict]]  # ResponsesInput: Text | Messages
    previous_response_id: Optional[str] = None
    tools: list[dict] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    system: Optional[str] = None
    instructions: Optional[str] = None


@dataclass
class ContentPart:
    type: str  # "output_text" etc.
    text: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        return d


@dataclass
class ResponsesOutputItem:
    type: str
    role: str
    content: list[ContentPart] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "role": self.role,
            "content": [c.to_dict() for c in self.content],
        }


@dataclass
class ResponsesUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ResponsesResponse:
    id: str
    object: str = "response"
    model: str = ""
    output: list[ResponsesOutputItem] = field(default_factory=list)
    usage: ResponsesUsage = field(default_factory=ResponsesUsage)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "output": [o.to_dict() for o in self.output],
            "usage": self.usage.to_dict(),
        }


# ── Chat Completions (outbound to provider) ─────────────────────────────────

@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class ChatRequest:
    model: str
    messages: list[ChatMessage] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False

    def to_dict(self) -> dict:
        d: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "stream": self.stream,
        }
        if self.tools:
            d["tools"] = self.tools
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        return d


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatResponse:
    choices: list[ChatChoice] = field(default_factory=list)
    usage: Optional[ChatUsage] = None


# ── SSE streaming types ─────────────────────────────────────────────────────

@dataclass
class DeltaFunction:
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class DeltaToolCall:
    index: int = 0
    id: Optional[str] = None
    function: Optional[DeltaFunction] = None


@dataclass
class ChatDelta:
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[DeltaToolCall]] = None


@dataclass
class ChatStreamChoice:
    delta: ChatDelta = field(default_factory=ChatDelta)
    finish_reason: Optional[str] = None


@dataclass
class ChatStreamChunk:
    choices: list[ChatStreamChoice] = field(default_factory=list)
    usage: Optional[ChatUsage] = None


def parse_chat_stream_chunk(data: str) -> Optional[ChatStreamChunk]:
    """Parse a JSON string into a ChatStreamChunk, returning None on failure."""
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None

    choices = []
    for c in obj.get("choices", []):
        delta_obj = c.get("delta", {})
        tool_calls = None
        if "tool_calls" in delta_obj and delta_obj["tool_calls"] is not None:
            tool_calls = []
            for tc in delta_obj["tool_calls"]:
                func = None
                if tc.get("function"):
                    func = DeltaFunction(
                        name=tc["function"].get("name"),
                        arguments=tc["function"].get("arguments"),
                    )
                tool_calls.append(
                    DeltaToolCall(
                        index=tc.get("index", 0),
                        id=tc.get("id"),
                        function=func,
                    )
                )
        delta = ChatDelta(
            role=delta_obj.get("role"),
            content=delta_obj.get("content"),
            reasoning_content=delta_obj.get("reasoning_content"),
            tool_calls=tool_calls,
        )
        choices.append(
            ChatStreamChoice(
                delta=delta,
                finish_reason=c.get("finish_reason"),
            )
        )

    usage = None
    if "usage" in obj and obj["usage"] is not None:
        u = obj["usage"]
        usage = ChatUsage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )

    return ChatStreamChunk(choices=choices, usage=usage)
