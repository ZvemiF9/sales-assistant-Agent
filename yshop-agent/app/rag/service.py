"""Day-three extractive demo answers; no LLM or shop execution hooks."""

from pydantic import ValidationError

from app.errors import ShopError
from app.rag.config import RagSettings
from app.rag.embeddings import Embeddings, LocalEmbeddings
from app.rag.evidence import select_primary_evidence
from app.rag.schemas import PolicyAnswer, PolicyQuery, PolicySource
from app.rag.store import PolicyStore


DEMO_NOTICE = "以下仅为商城演示规则，不构成真实商城承诺；实际安排请通过商城已有客服入口确认。"


def normalize_question(question: str) -> str:
    try:
        return PolicyQuery(question=question).question
    except ValidationError:
        raise ShopError("invalid_request") from None


class PolicyService:
    def __init__(self, settings: RagSettings | None = None, *, embeddings: Embeddings | None = None):
        self.settings = settings if settings is not None else RagSettings()
        self._store = PolicyStore(self.settings, embeddings if embeddings is not None else LocalEmbeddings(self.settings))

    def search_policy(self, question: str, *, top_k: int = 4) -> list[PolicySource]:
        return self._store.query(normalize_question(question), top_k=top_k)

    def answer(self, question: str) -> PolicyAnswer:
        question = normalize_question(question)
        candidates = self.search_policy(question, top_k=self.settings.top_k)
        primary = select_primary_evidence(question, candidates, self._store.embeddings,
                                          min_score=self.settings.evidence_min_score)
        sources = [primary] if primary is not None else []
        if not sources:
            return PolicyAnswer(question=question, status="no_evidence", sources=[], notice=DEMO_NOTICE,
                                answer="演示资料中未找到足够依据，请通过商城已有客服入口咨询。")
        excerpts = "\n\n".join(f"[{index}] {source.text}" for index, source in enumerate(sources, 1))
        return PolicyAnswer(question=question, status="answered", sources=sources, notice=DEMO_NOTICE,
                            answer="以下为检索到的演示规则原文摘录：\n\n" + excerpts)
