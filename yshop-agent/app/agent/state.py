"""Only completed, public conversation data may enter a session snapshot."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from app.schemas import PositiveId, ProductDetail, ProductSearchInput, PublicModel


MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARACTERS = 4000
MAX_SNAPSHOT_BYTES = 65536


class PublicMessage(PublicModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_MESSAGE_CHARACTERS)]


class ConversationState(PublicModel):
    history: list[PublicMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    conditions: ProductSearchInput = Field(default_factory=ProductSearchInput)
    candidates: list[ProductDetail] = Field(default_factory=list, max_length=3)
    selected_product_id: PositiveId | None = None
    selected_sku_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                                    min_length=1, max_length=128)] | None = None
    pending_action_id: Annotated[str, StringConstraints(strict=True, min_length=32, max_length=32,
                                                        pattern=r"^[0-9a-f]{32}$")] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def bounded_snapshot(self) -> "ConversationState":
        if len(self.model_dump_json().encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise ValueError("conversation snapshot exceeds size limit")
        return self
