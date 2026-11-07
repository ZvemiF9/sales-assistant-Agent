from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


PositiveId = Annotated[int, Field(strict=True, gt=0)]
NonBlankString = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Stock = Annotated[int, Field(strict=True, ge=0)]


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class UserIdentity(PublicModel):
    user_id: PositiveId
    nickname: str = ""


class ProductSKU(PublicModel):
    name: str
    unique_id: NonBlankString
    price: Money
    stock: Stock
    image: str


class ProductDetail(PublicModel):
    product_id: PositiveId
    name: NonBlankString
    description: str
    image: str
    skus: list[ProductSKU]


class ProductSearchInput(PublicModel):
    keyword: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    category_id: Annotated[int, Field(gt=0, le=9223372036854775807)] | None = None
    min_price: Money | None = None
    max_price: Money | None = None
    page: Annotated[int, Field(ge=1, le=2147483647)] = 1
    limit: Annotated[int, Field(ge=1, le=20)] = 10

    @field_validator("category_id", "page", "limit", mode="before")
    @classmethod
    def integer_query_value(cls, value):
        # HTTP query strings can be digits; booleans and float coercion cannot be IDs.
        if value is not None and not (
            type(value) is int or (isinstance(value, str) and value.isascii() and value.isdecimal())
        ):
            raise ValueError("expected an integer")
        return value

    @model_validator(mode="after")
    def ordered_budget(self) -> "ProductSearchInput":
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("invalid budget interval")
        return self


class ProductSearchResult(PublicModel):
    products: list[ProductDetail]
    page: PositiveId
    limit: Annotated[int, Field(strict=True, ge=1, le=20)]
    total: Annotated[int, Field(strict=True, ge=0)]


class CartAddInput(PublicModel):
    """Validated input for an ordinary-product cart write."""

    product_id: PositiveId
    unique_id: NonBlankString
    quantity: Annotated[int, Field(strict=True, ge=1, le=9999)] = 1


class CartItem(PublicModel):
    cart_id: PositiveId
    product_id: PositiveId
    unique_id: NonBlankString | None
    quantity: Annotated[int, Field(strict=True, gt=0)]
    name: NonBlankString
    sku_name: str
    image: str
    price: Money | None
    stock: Stock | None


class CartSnapshot(PublicModel):
    valid: list[CartItem]
    invalid: list[CartItem]


class CartAddResult(PublicModel):
    cart_id: PositiveId
    status: Literal["succeeded"] = "succeeded"
