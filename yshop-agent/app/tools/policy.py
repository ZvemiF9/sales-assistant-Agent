"""Read-only policy tool: accepts no authorization or shop client."""

from starlette.concurrency import run_in_threadpool

from app.rag.schemas import PolicySource
from app.rag.service import PolicyService


async def search_policy(service: PolicyService, question: str) -> list[PolicySource]:
    return await run_in_threadpool(service.search_policy, question)
