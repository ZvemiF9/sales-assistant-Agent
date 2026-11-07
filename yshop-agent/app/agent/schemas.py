"""Public chat data, using existing product and policy evidence allowlists."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.rag.schemas import PolicySource
from app.schemas import ProductDetail, PublicModel


NonNegativeInteger = Annotated[int, Field(strict=True, ge=0)]
SessionId = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{32}$")]


class ChatRequest(PublicModel):
    message: Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                              min_length=1, max_length=1000)]
    session_id: SessionId | None = None


class ToolTrace(PublicModel):
    name: Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    status: Literal["succeeded", "failed"]
    elapsed_ms: NonNegativeInteger


class TokenUsage(PublicModel):
    input_tokens: NonNegativeInteger | None = None
    output_tokens: NonNegativeInteger | None = None


class ChatResponse(PublicModel):
    answer: Annotated[str, Field(strict=True)]
    status: Literal["answered", "needs_input", "no_evidence", "limit_reached"]
    products: Annotated[list[ProductDetail], Field(max_length=3)] = Field(default_factory=list)
    sources: list[PolicySource] = Field(default_factory=list)
    notice: Annotated[str, Field(strict=True)] | None = None
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    model_rounds: Annotated[int, Field(strict=True, ge=1, le=5)]
    usage: TokenUsage = Field(default_factory=TokenUsage)


class SessionChatResponse(ChatResponse):
    session_id: SessionId
    status: Literal[
        "answered", "needs_input", "no_evidence", "limit_reached",
        "awaiting_confirmation", "executing", "succeeded", "failed", "unknown", "cancelled",
    ]
    model_rounds: Annotated[int, Field(strict=True, ge=0, le=5)]
    pending_action: "CartActionCard | None" = None
    action_result: "CartActionResult | None" = None


from app.actions.schemas import CartActionCard, CartActionResult

SessionChatResponse.model_rebuild()
