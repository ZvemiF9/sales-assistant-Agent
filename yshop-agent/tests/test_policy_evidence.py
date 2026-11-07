"""Sentence evidence boundaries; deterministic vectors are not model acceptance."""

import math

import pytest
from pydantic import ValidationError

from app.errors import ShopError
from app.rag.config import RagSettings
from app.rag.schemas import PolicySource
from app.rag.service import PolicyService


QUESTION = "这个演示安排适用吗？"


def vector(cosine, scale=1.0):
    return [cosine * scale, math.sqrt(1 - cosine ** 2) * scale, 0.0]


class SentenceEmbeddings:
    dimension = 3

    def __init__(self, sentences, *, query_vector=None):
        self.sentences = sentences
        self.query_vector = [1.0, 0.0, 0.0] if query_vector is None else query_vector
        self.query_calls = []
        self.document_calls = []

    def fingerprint(self):
        return {"model": "deterministic-evidence-test-only"}

    def embed_query(self, text):
        self.query_calls.append(text)
        return self.query_vector

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self.sentences[text] for text in texts]


def source(text, *, chunk_id="chunk-a", score=0.70):
    return PolicySource(document_id="arbitrary-document-" + chunk_id, title="演示资料",
                        section="测试章节", source="knowledge/policies/example.md",
                        version="demo-test", kind="demo", chunk_id=chunk_id,
                        text=text, score=score)


def service_with_candidates(monkeypatch, candidates, sentences, *, query_vector=None, **settings):
    model = SentenceEmbeddings(sentences, query_vector=query_vector)
    service = PolicyService(RagSettings(_env_file=None, **settings), embeddings=model)
    calls = []

    def candidates_query(question, *, top_k):
        calls.append((question, top_k))
        return candidates

    monkeypatch.setattr(service._store, "query", candidates_query)
    return service, model, calls


def test_answer_suppresses_unrelated_candidates_and_preserves_public_original(monkeypatch):
    primary = source("\n演示规则：该安排适用。 \n", score=0.73123456789)
    unrelated = source("另一个主题的原文。", chunk_id="chunk-b", score=0.71)
    service, _, calls = service_with_candidates(monkeypatch, [primary, unrelated],
                                               {"演示规则：该安排适用。": vector(0.9),
                                                "另一个主题的原文。": vector(0.3),
                                                "演示规则：该安排适用。 ": vector(0.9)})
    result = service.answer("  " + QUESTION + "  ")
    assert result.status == "answered"
    assert result.sources == [primary]
    assert result.sources[0].model_dump() == primary.model_dump()
    assert primary.text in result.answer and unrelated.text not in result.answer
    assert calls == [(QUESTION, 4)]


def test_answer_reranks_second_raw_hit_but_search_policy_retains_candidates(monkeypatch):
    first = source("相近但较弱的原文。", score=0.84)
    second = source("最切题的原文。", chunk_id="chunk-b", score=0.62)
    service, model, _ = service_with_candidates(monkeypatch, [first, second],
                                               {first.text: vector(0.61), second.text: vector(0.99)})
    assert service.search_policy(QUESTION) == [first, second]
    assert model.query_calls == [] and model.document_calls == []
    assert service.answer(QUESTION).sources == [second]


def test_mixed_rank_uses_chunk_score_as_well_as_sentence_cosine(monkeypatch):
    first = source("完整背景更吻合的原文。", score=0.90)
    second = source("单句稍微更接近的原文。", chunk_id="chunk-b", score=0.54)
    service, _, _ = service_with_candidates(monkeypatch, [first, second],
                                           {first.text: vector(0.75), second.text: vector(0.90)})
    assert service.answer(QUESTION).sources == [first]


def test_low_sentence_evidence_refuses_even_with_high_chunk_score(monkeypatch):
    candidate = source("只谈别的安排。", score=0.99)
    service, _, _ = service_with_candidates(monkeypatch, [candidate], {candidate.text: vector(0.59)})
    result = service.answer(QUESTION)
    assert result.status == "no_evidence" and result.sources == []
    assert "未找到" in result.answer and "客服" in result.answer


def test_no_candidates_does_not_load_or_call_embeddings(monkeypatch):
    service, model, _ = service_with_candidates(monkeypatch, [], {})
    assert service.answer(QUESTION).status == "no_evidence"
    assert model.query_calls == [] and model.document_calls == []


def test_best_missing_sentence_refuses_without_falling_back_to_other_candidate(monkeypatch):
    missing = source("本资料未说明具体安排。", score=0.8)
    weaker = source("可在页面查看一般提示。", chunk_id="chunk-b", score=0.7)
    service, _, _ = service_with_candidates(monkeypatch, [missing, weaker],
                                           {missing.text: vector(0.9), weaker.text: vector(0.8)})
    assert service.answer("具体安排是什么？").status == "no_evidence"


@pytest.mark.parametrize("text", ["本资料没有具体时限。", "文档未提供具体时限。",
                                  "资料不包含所需金额。", "这份资料没有统一承诺。"])
def test_generic_missing_detail_statement_is_no_evidence(monkeypatch, text):
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate], {text: vector(0.9)})
    assert service.answer("具体数值是多少？").status == "no_evidence"


MIXED = "本资料没有固定的银行到账天数，不保证即时到账，亦不自动发起退款。"


@pytest.mark.parametrize("question", ["商家收到退回的货了，钱是不是马上到账？", "钱会不会立即到账？",
                                      "钱能否立即到账？", "钱能不能立即到账？", "钱立即到账吗？"])
def test_boolean_question_can_use_separate_matching_negative_assertion(monkeypatch, question):
    candidate = source(MIXED)
    service, model, _ = service_with_candidates(monkeypatch, [candidate],
        {MIXED: vector(0.9), "不保证即时到账": vector(0.85), "亦不自动发起退款。": vector(0.3)})
    result = service.answer(question)
    assert result.status == "answered" and result.sources == [candidate]
    assert any("不保证即时到账" in batch for batch in model.document_calls)


@pytest.mark.parametrize("question", ["银行卡退款保证几个自然日到账？",
                                      "退款是否已提交需核实；银行卡退款保证几个自然日到账？",
                                      "退款是否保证几个自然日到账？"])
def test_open_numeric_request_cannot_use_boolean_exception(monkeypatch, question):
    candidate = source(MIXED)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {MIXED: vector(0.9), "不保证即时到账": vector(0.9), "亦不自动发起退款。": vector(0.8)})
    assert service.answer(question).status == "no_evidence"


@pytest.mark.parametrize("question", ["银行卡退款几个自然日到账，可以告诉我吗？",
                                      "银行卡退款几个自然日到账；能否说明一下？"])
def test_polite_request_tail_inherits_open_detail_question(monkeypatch, question):
    candidate = source(MIXED)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {MIXED: vector(0.9), "不保证即时到账": vector(0.9), "亦不自动发起退款。": vector(0.8)})
    result = service.answer(question)
    assert result.status == "no_evidence" and result.sources == []


def test_open_background_does_not_override_final_factual_boolean_question(monkeypatch):
    candidate = source(MIXED)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {MIXED: vector(0.9), "不保证即时到账": vector(0.9), "亦不自动发起退款。": vector(0.8)})
    result = service.answer("不知道几天会处理，钱是不是马上到账？")
    assert result.status == "answered" and result.sources == [candidate]


def test_boolean_in_context_does_not_override_final_nonboolean_request(monkeypatch):
    candidate = source(MIXED)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {MIXED: vector(0.9), "不保证即时到账": vector(0.9), "亦不自动发起退款。": vector(0.8)})
    assert service.answer("退款是否已提交还需核实，接下来请介绍到账安排？").status == "no_evidence"


def test_boolean_exception_requires_assertion_to_clear_independent_threshold(monkeypatch):
    candidate = source(MIXED)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {MIXED: vector(0.9), "不保证即时到账": vector(0.59), "亦不自动发起退款。": vector(0.4)})
    assert service.answer("钱是否立即到账？").status == "no_evidence"


def test_missing_context_and_customer_referral_are_not_independent_assertions(monkeypatch):
    text = "具体补货日期，本资料没有统一承诺，需通过商城已有客服入口了解。"
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {text: vector(0.9), "具体补货日期": vector(0.95), "需通过商城已有客服入口了解。": vector(0.95)})
    assert service.answer("是否有固定补货日期？").status == "no_evidence"


def test_boolean_exception_accepts_generic_affirmative_assertion(monkeypatch):
    text = "文档未说明处理天数；确认通过后会提交处理。"
    candidate = source(text)
    service, model, _ = service_with_candidates(monkeypatch, [candidate],
        {text: vector(0.9), "确认通过后会提交处理。": vector(0.85)})
    assert service.answer("确认通过后会不会提交处理？").status == "answered"
    assert any("确认通过后会提交处理。" in batch for batch in model.document_calls)


def test_valid_negative_business_assertion_is_not_missing_information(monkeypatch):
    text = "把商品加入购物车不代表锁定库存；下单时仍需重新校验。"
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate], {text: vector(0.9)})
    assert service.answer("加入购物车会锁定库存吗？").sources == [candidate]


def test_document_attribution_does_not_turn_product_exclusion_into_missing_information(monkeypatch):
    text = "文档说明：该商品不包含电池。"
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate], {text: vector(0.9)})
    assert service.answer("商品是否包含电池？").sources == [candidate]


def test_inherited_missing_document_clause_is_not_an_independent_assertion(monkeypatch):
    text = "这份文档仅说明规则，不包含具体时限。"
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {text: vector(0.9), "不包含具体时限。": vector(0.95)})
    assert service.answer("文档是否包含具体时限？").status == "no_evidence"


@pytest.mark.parametrize("text", ["这份文档仅说明规则，供用户参考，不包含具体时限。",
                                  "这份文档仅说明规则；供用户参考；未提供具体时限。"])
def test_missing_document_subject_survives_intervening_subordinate_clause(monkeypatch, text):
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
                                           {text: vector(0.9), "不包含具体时限。": vector(0.95)})
    result = service.answer("文档是否包含具体时限？")
    assert result.status == "no_evidence" and result.sources == []


def test_explicit_new_subject_stops_missing_document_inheritance(monkeypatch):
    text = "这份文档仅说明规则，该商品尺寸较小，不包含电池。"
    candidate = source(text)
    service, _, _ = service_with_candidates(monkeypatch, [candidate], {text: vector(0.9)})
    result = service.answer("商品是否包含电池？")
    assert result.status == "answered" and result.sources == [candidate]
    assert result.sources[0].text == text and result.sources[0].score == candidate.score


def test_sentence_embedding_preserves_original_spans_and_comma_semicolon_context(monkeypatch):
    text = "原始甲，保留；条件。 原始乙？\n原始丙！末尾"
    candidate = source(text)
    spans = ["原始甲，保留；条件。", " 原始乙？", "原始丙！", "末尾"]
    service, model, _ = service_with_candidates(monkeypatch, [candidate],
                                               {span: vector(0.9) for span in spans})
    assert service.answer(QUESTION).sources == [candidate]
    assert model.document_calls == [spans]
    assert model.query_calls == [QUESTION]


def test_equal_rank_tie_is_stable_by_chunk_id(monkeypatch):
    first = source("排序乙。", chunk_id="chunk-b")
    second = source("排序甲。", chunk_id="chunk-a")
    service, _, _ = service_with_candidates(monkeypatch, [first, second],
                                           {first.text: vector(0.9), second.text: vector(0.9)})
    assert service.answer(QUESTION).sources == [second]


def test_sentence_cosine_normalizes_non_unit_vectors(monkeypatch):
    first = source("很大的弱向量。", score=0.7)
    second = source("很小的强向量。", chunk_id="chunk-b", score=0.7)
    service, _, _ = service_with_candidates(monkeypatch, [first, second],
        {first.text: vector(0.3, 100.0), second.text: vector(0.8, 0.001)}, query_vector=[3.0, 0.0, 0.0])
    assert service.answer(QUESTION).sources == [second]


@pytest.mark.parametrize("bad", [[1.0, 0.0], [float("nan"), 0.0, 1.0], [float("inf"), 0.0, 1.0],
                                 [0.0, 0.0, 0.0], [True, 0.0, 0.0], ["private-vector", 0.0, 1.0]])
@pytest.mark.parametrize("location", ["query", "sentence"])
def test_invalid_evidence_vectors_raise_sanitized_error(monkeypatch, bad, location):
    candidate = source("原文。")
    service, _, _ = service_with_candidates(monkeypatch, [candidate],
        {candidate.text: bad if location == "sentence" else vector(0.9)},
        query_vector=bad if location == "query" else None)
    with pytest.raises(ShopError) as caught:
        service.answer(QUESTION)
    assert caught.value.code == "rag_unavailable" and caught.value.http_status == 503
    assert "private-vector" not in str(caught.value) and QUESTION not in str(caught.value)


def test_evidence_threshold_is_separate_config_and_honored(monkeypatch):
    settings = RagSettings(_env_file=None)
    assert settings.evidence_min_score == 0.60
    assert settings.min_score == 0.53 and settings.top_k == 4
    text = "匹配程度介于两个证据门槛之间。"
    candidate = source(text, score=0.99)
    service, _, _ = service_with_candidates(monkeypatch, [candidate], {text: vector(0.7)},
                                           evidence_min_score=0.75)
    assert service.answer(QUESTION).status == "no_evidence"


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 1.01, -1.01])
def test_invalid_evidence_threshold_is_rejected(threshold):
    with pytest.raises(ValidationError):
        RagSettings(_env_file=None, evidence_min_score=threshold)
