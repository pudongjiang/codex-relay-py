import hashlib
import threading
import uuid

from .types import ChatMessage


def _content_key(content: str) -> int:
    """Hash assistant message content for turn-level reasoning lookup."""
    h = hashlib.sha256(content.encode()).hexdigest()
    # Use first 16 chars of hex digest as u64-equivalent key
    return int(h[:16], 16)


class SessionStore:
    """In-memory session storage with reasoning_content indexes.

    Maps response_id -> accumulated message history for each session.
    Codex uses ``previous_response_id`` to continue a conversation; we maintain
    the full messages[] here so each Chat Completions call is self-contained.

    Also maintains call_id -> reasoning_content so that thinking-capable models
    can have their reasoning_content round-tripped back when Codex replays
    tool-call history in subsequent requests.

    For assistant messages without tool calls (pure text), reasoning_content
    is indexed by a fingerprint of the assistant content, so it can be
    recovered when Codex replays the full conversation in ``input`` without
    using ``previous_response_id``.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._inner: dict[str, list[ChatMessage]] = {}
        self._reasoning: dict[str, str] = {}
        self._turn_reasoning: dict[int, str] = {}

    def store_reasoning(self, call_id: str, reasoning: str) -> None:
        """Store reasoning_content keyed by the tool call_id."""
        if reasoning:
            with self._lock:
                self._reasoning[call_id] = reasoning

    def get_reasoning(self, call_id: str) -> Optional[str]:
        """Look up stored reasoning_content for a call_id."""
        with self._lock:
            return self._reasoning.get(call_id)

    def store_turn_reasoning(
        self,
        prior: list[ChatMessage],
        assistant: ChatMessage,
        reasoning: str,
    ) -> None:
        """Store reasoning_content for an assistant turn by content fingerprint and call_ids."""
        if not reasoning:
            return
        with self._lock:
            content = assistant.content or ""
            if content:
                key = _content_key(content)
                self._turn_reasoning[key] = reasoning
            # Also store under each tool call_id
            if assistant.tool_calls:
                for tc in assistant.tool_calls:
                    call_id = tc.get("id", "")
                    if call_id:
                        self._reasoning[call_id] = reasoning

    def get_turn_reasoning(
        self,
        prior: list[ChatMessage],
        assistant: ChatMessage,
    ) -> Optional[str]:
        """Look up reasoning_content for an assistant turn by its text content."""
        content = assistant.content or ""
        if not content:
            return None
        key = _content_key(content)
        with self._lock:
            return self._turn_reasoning.get(key)

    def get_history(self, response_id: str) -> list[ChatMessage]:
        """Retrieve history for a prior response_id, or empty list if not found."""
        with self._lock:
            return list(self._inner.get(response_id, []))

    def new_id(self) -> str:
        """Allocate a fresh response_id without storing anything yet."""
        return f"resp_{uuid.uuid4().hex}"

    def save_with_id(self, id: str, messages: list[ChatMessage]) -> None:
        """Store under a pre-allocated response_id (streaming path)."""
        with self._lock:
            self._inner[id] = list(messages)

    def save(self, messages: list[ChatMessage]) -> str:
        """Allocate an id and store atomically (non-streaming path)."""
        id = self.new_id()
        with self._lock:
            self._inner[id] = list(messages)
        return id
