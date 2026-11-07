"""Strict, bounded public data for immutable cart confirmations."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from app.errors import ShopError
from app.schemas import PublicModel


ACTION_TTL_SECONDS = 900
CART_PATH = "/pages/shop/ShoppingCart/index"
ActionId = Annotated[str, StringConstraints(strict=True, min_length=32, max_length=32,
                                           pattern=r"^[0-9a-f]{32}$")]
BoundedId = Annotated[int, Field(strict=True, gt=0, le=9223372036854775807)]
ActionStatus = Literal["pending", "executing", "succeeded", "failed", "unknown", "cancelled"]
ResultStatus = Literal["executing", "succeeded", "failed", "unknown", "cancelled"]
ActionResponseStatus = Literal[
    "awaiting_confirmation", "needs_input", "executing", "succeeded", "failed", "unknown", "cancelled",
]
Name = Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                       min_length=1, max_length=500)]
Price = Annotated[Decimal, Field(ge=0, allow_inf_nan=False, max_digits=16, decimal_places=2)]
TotalPrice = Annotated[Decimal, Field(ge=0, allow_inf_nan=False, max_digits=20, decimal_places=2)]


class CartActionCard(PublicModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["add_to_cart"] = "add_to_cart"
    action_id: ActionId
    session_id: ActionId
    product_id: BoundedId
    product_name: Name
    unique_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                               min_length=1, max_length=128)]
    sku_name: Name
    quantity: Annotated[int, Field(strict=True, ge=1, le=9999)]
    unit_price: Price
    total_price: TotalPrice
    image: Annotated[str, StringConstraints(strict=True, max_length=2048)]
    expires_at: Annotated[float, Field(strict=True, gt=0, le=253402300799, allow_inf_nan=False)]

    @model_validator(mode="after")
    def correct_total(self) -> "CartActionCard":
        if self.total_price != self.unit_price * self.quantity:
            raise ValueError("action total does not match price and quantity")
        return self


class CartActionResult(PublicModel):
    model_config = ConfigDict(frozen=True)

    action_id: ActionId
    session_id: ActionId
    status: ResultStatus
    cart_id: BoundedId | None = None
    error_code: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64,
                                                pattern=r"^[a-z][a-z0-9_]*$")] | None = None
    cart_path: Literal["/pages/shop/ShoppingCart/index"] = CART_PATH

    @field_validator("error_code")
    @classmethod
    def known_public_error(cls, value: str | None) -> str | None:
        if value is not None and ShopError(value).code != value:
            raise ValueError("unknown public error code")
        return value

    @model_validator(mode="after")
    def coherent_result(self) -> "CartActionResult":
        if (self.status == "succeeded") != (self.cart_id is not None):
            raise ValueError("only succeeded results carry a cart ID")
        if self.status in {"succeeded", "executing"} and self.error_code is not None:
            raise ValueError("result status cannot carry an error")
        return self


class ActionDecision(PublicModel):
    decision: Literal["confirm", "cancel"]


class ActionResponse(PublicModel):
    session_id: ActionId
    status: ActionResponseStatus
    answer: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4000)]
    pending_action: CartActionCard | None = None
    action_result: CartActionResult | None = None

    @model_validator(mode="after")
    def coherent_response(self) -> "ActionResponse":
        if self.status == "awaiting_confirmation":
            if self.pending_action is None or self.action_result is not None:
                raise ValueError("confirmation response requires only a pending card")
        elif self.status == "needs_input":
            if self.pending_action is not None or self.action_result is not None:
                raise ValueError("input response cannot carry an action")
        elif (self.pending_action is not None or self.action_result is None
              or self.action_result.status != self.status):
            raise ValueError("outcome response requires a matching result")
        for value in (self.pending_action, self.action_result):
            if value is not None and value.session_id != self.session_id:
                raise ValueError("response session does not match its action")
        return self
