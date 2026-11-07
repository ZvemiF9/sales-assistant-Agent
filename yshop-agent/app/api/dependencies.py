import re
from dataclasses import dataclass, field

from fastapi import Request

from app.errors import ShopError
from app.schemas import UserIdentity


_BEARER = re.compile(r"Bearer [A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    authorization: str = field(repr=False)
    user: UserIdentity


async def require_identity(request: Request) -> RequestIdentity:
    headers = request.headers.getlist("authorization")
    if len(headers) != 1 or _BEARER.fullmatch(headers[0]) is None:
        raise ShopError("login_required")
    authorization = headers[0]
    user = await request.app.state.shop.get_user(authorization)
    return RequestIdentity(authorization=authorization, user=user)
