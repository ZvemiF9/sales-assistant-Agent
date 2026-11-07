from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.schemas import PublicModel


PolicyQuestion = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=1000)]


class PolicyQuery(PublicModel):
    question: PolicyQuestion


class PolicySource(PublicModel):
    document_id: str
    title: str
    section: str
    source: str
    version: str
    kind: Literal["demo"]
    chunk_id: str
    # Preserve exact original chunk text, including internal boundary whitespace.
    text: str
    score: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class PolicyAnswer(PublicModel):
    question: PolicyQuestion
    answer: str
    status: Literal["answered", "no_evidence"]
    sources: list[PolicySource]
    notice: str
