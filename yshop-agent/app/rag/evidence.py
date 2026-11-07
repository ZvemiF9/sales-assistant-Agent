"""Transient sentence evidence over verified vector candidates.

Ranking uses the same injected embeddings as the index. Syntax checks only
classify missing-information statements and a narrow boolean exception;
they do not retrieve or route questions to policy topics.
"""

import math
import re

from app.errors import ShopError
from app.rag.embeddings import Embeddings
from app.rag.schemas import PolicySource
from app.rag.store import _vector


CHUNK_WEIGHT = 0.5
SENTENCE_WEIGHT = 0.5
MAX_SOURCES = 1

_SENTENCES = re.compile(r"[^。！？.!?\r\n]+[。！？.!?]*|[。！？.!?]+")
_DOCUMENT = re.compile(r"资料|文档|知识库")
_ABSENCE_WORDS = r"(?:没有|未(?:提供|说明|给出|列出)|不(?:包含|提供|含|承诺))"
_MISSING = re.compile(
    r"(?:资料|文档|知识库)[^，,；;：:。！？.!?\r\n]*?" + _ABSENCE_WORDS
)
_INHERITED_ABSENCE = re.compile(r"^\s*(?:也|亦|并|且)?" + _ABSENCE_WORDS)
_SUBORDINATE_CONTINUATION = re.compile(r"^\s*(?:仅|也|亦|并|且|可)?(?:供|便于|用于|用来|以便|作为)")
_BOOLEAN = re.compile(r"吗|是否|是不是|能否|会不会|能不能")
_OPEN_QUESTION = re.compile(r"多少|几|多久|何时|什么|哪|怎么|如何")
_POLITE_REQUEST = re.compile(
    r"(?:请|请问|麻烦)?\s*(?:可以|能否|能不能|可否|是否(?:能|可以)?|能|会不会)?\s*"
    r"(?:(?:给|向|为|跟)我)?(?:告诉|告知|说明|解释|回答|介绍|说)"
    r"(?:我)?(?:一下|下)?(?:可以|好|行)?(?:吗)?"
)
_REFERRAL = re.compile(r"客服|咨询|询问|联系|查看.*(?:页面|说明)|确认.*入口")
_ASSERTION = re.compile(
    r"不|未|无法|无须|无需|可(?:以|能)?|能|会|应(?:当|该)?|需(?:要)?|必须|"
    r"支持|允许|保证|代表|等于|属于|为|是|由|按|以|取决于"
)


def _unit_vector(value, dimension: int) -> list[float]:
    values = _vector(value, dimension)
    # Rescaling before computing the norm also handles very large/small finite
    # vectors without overflow or underflow changing their cosine.
    scale = max(abs(number) for number in values)
    scaled = [number / scale for number in values]
    norm = math.sqrt(math.fsum(number * number for number in scaled))
    return [number / norm for number in scaled]


def _cosine(left: list[float], right: list[float]) -> float:
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


def _sentence_spans(text: str) -> list[str]:
    # Every embedded span is an unchanged substring of the original chunk.
    # Commas and semicolons remain inside sentence vectors for context.
    return [match.group() for match in _SENTENCES.finditer(text) if match.group().strip()]


def _boolean_question(question: str) -> bool:
    # Generic polite request tails inherit the preceding asked fact; their
    # modal marker alone cannot turn an open detail request into a claim.
    # A final factual boolean clause keeps its own role despite open background.
    sentence = re.split(r"[。！？.!?\r\n]", question.rstrip("。！？.!?\r\n "))[-1]
    clauses = [clause.strip() for clause in re.split(r"[，,；;]", sentence) if clause.strip()]
    while clauses and _POLITE_REQUEST.fullmatch(clauses[-1]):
        clauses.pop()
    if not clauses:
        return False
    interrogative = clauses[-1]
    return _BOOLEAN.search(interrogative) is not None and _OPEN_QUESTION.search(interrogative) is None


def _missing_statement(sentence: str) -> bool:
    document_subject = False
    for clause in re.split(r"[，,；;]", sentence):
        if _MISSING.search(clause) or (document_subject and _INHERITED_ABSENCE.search(clause)):
            return True
        document = _DOCUMENT.search(clause)
        # A colon after the document reference introduces attributed content,
        # whose exclusions can describe a product rather than absent material.
        if document is not None:
            document_subject = not re.search(r"[：:]", clause[document.end():])
        elif re.search(r"[：:]", clause):
            document_subject = False
        elif not _SUBORDINATE_CONTINUATION.search(clause):
            # Keep an omitted document subject through known subordinate
            # continuations; a separate clause can introduce a new subject.
            document_subject = False
    return False


def _independent_assertions(sentence: str) -> list[str]:
    assertions = []
    for clause in re.split(r"[，,；;]", sentence):
        if (_DOCUMENT.search(clause) or _INHERITED_ABSENCE.search(clause) or _REFERRAL.search(clause)
                or _BOOLEAN.search(clause) or not _ASSERTION.search(clause)):
            continue
        assertions.append(clause)
    return assertions


def select_primary_evidence(
    question: str, candidates: list[PolicySource], embeddings: Embeddings, *, min_score: float
) -> PolicySource | None:
    """Return one unchanged chunk, or refuse when its best sentence lacks evidence."""
    if not candidates:
        return None
    try:
        sentences_by_source = [_sentence_spans(source.text) for source in candidates]
        sentences = [sentence for spans in sentences_by_source for sentence in spans]
        if not sentences:
            return None
        question_vector = _unit_vector(embeddings.embed_query(question), embeddings.dimension)
        sentence_vectors = list(embeddings.embed_documents(sentences))
        if len(sentence_vectors) != len(sentences):
            raise ValueError
        sentence_scores = [_cosine(question_vector, _unit_vector(value, embeddings.dimension))
                           for value in sentence_vectors]
        ranked = []
        offset = 0
        for source, spans in zip(candidates, sentences_by_source, strict=True):
            if spans:
                scores = sentence_scores[offset:offset + len(spans)]
                best = max(range(len(spans)), key=scores.__getitem__)
                ranked.append((CHUNK_WEIGHT * source.score + SENTENCE_WEIGHT * scores[best],
                               source, scores[best], spans[best]))
            offset += len(spans)
        _, primary, best_score, best_sentence = min(ranked, key=lambda row: (-row[0], row[1].chunk_id))
        if best_score < min_score:
            return None
        if _missing_statement(best_sentence):
            if not _boolean_question(question):
                return None
            assertions = _independent_assertions(best_sentence)
            if not assertions:
                return None
            assertion_vectors = list(embeddings.embed_documents(assertions))
            if len(assertion_vectors) != len(assertions):
                raise ValueError
            scores = [_cosine(question_vector, _unit_vector(value, embeddings.dimension))
                      for value in assertion_vectors]
            if max(scores) < min_score:
                return None
        return primary
    except ShopError:
        raise
    except Exception:
        raise ShopError("rag_unavailable") from None
