"""Real Chroma boundary tests; deterministic vectors are not model acceptance."""

from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from app.errors import ShopError
from app.rag.documents import load_policy_chunks


def rag_module(name):
    assert importlib.util.find_spec("app.rag." + name) is not None, f"RAG {name} implementation is missing"
    return importlib.import_module("app.rag." + name)


class DeterministicEmbeddings:
    dimension = 3

    def __init__(self, *, identity="test-v1", vector=None):
        self.identity = identity
        self.vector = vector
        self.calls = []

    def fingerprint(self):
        return {"model": "deterministic-test-only", "dimension": self.dimension,
                "preprocessing": "exact-text-v1", "weights_sha256": self.identity,
                "versions": {"test": "1"}}

    def embed_documents(self, texts):
        self.calls.extend(texts)
        return [self.vector if self.vector is not None else [1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text):
        self.calls.append(text)
        return self.vector if self.vector is not None else ([0.0, 1.0, 0.0] if "医疗保险" in text else [1.0, 0.0, 0.0])


def write_policy(root, *, name="returns", text="演示规则：未使用商品可在签收后七天内申请退货。"):
    root.mkdir(parents=True, exist_ok=True)
    meta = {"document_id": name, "title": "商城演示规则", "source": f"knowledge/policies/{name}.md",
            "version": "demo-v1", "kind": "demo"}
    path = root / (name + ".md")
    path.write_text("---\n" + json.dumps(meta, ensure_ascii=False) + "\n---\n# 商城演示规则\n\n## 退货\n\n" + text + "\n", encoding="utf-8")
    return path


def setup_store(tmp_path, *, embedding=None):
    config = rag_module("config")
    store = rag_module("store")
    knowledge = tmp_path / "knowledge" / "policies"
    write_policy(knowledge)
    settings = config.RagSettings(_env_file=None, knowledge_dir=knowledge,
                                 chroma_dir=tmp_path / "chroma", model_cache_dir=tmp_path / "cache")
    embedding = embedding or DeterministicEmbeddings()
    return settings, embedding, store.PolicyStore(settings, embedding)


def test_real_chroma_persistence_reopen_preserves_original_text_and_metadata(tmp_path):
    settings, embedding, store = setup_store(tmp_path)
    original = load_policy_chunks(settings.knowledge_dir)[0]
    # Edge whitespace inside chunks is source data and must survive Pydantic.
    chunk = replace(original, text="\n" + original.text + " \n")
    # A build is normally fed the loader output; use a long paragraph for real edges.
    write_policy(settings.knowledge_dir, text="演示退货说明。" * 90)
    chunks = load_policy_chunks(settings.knowledge_dir)
    manifest = store.build(chunks)
    reopened = rag_module("store").PolicyStore(settings, DeterministicEmbeddings())
    hits = reopened.query("退货条件？", top_k=4)
    assert len(hits) == len(chunks)
    for hit in hits:
        expected = next(c for c in chunks if c.chunk_id == hit.chunk_id)
        assert hit.text == expected.text
        assert hit.model_dump(exclude={"text", "score"}) == expected.metadata
        assert hit.score == pytest.approx(1.0)
    source = rag_module("schemas").PolicySource(**chunk.metadata, text=chunk.text, score=0.9)
    assert source.text == chunk.text
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["model_fingerprint"] == embedding.fingerprint()


def test_repeat_build_is_identical_and_rebuild_removes_deleted_and_modified_chunks(tmp_path):
    settings, embedding, store = setup_store(tmp_path)
    extra = write_policy(settings.knowledge_dir, name="obsolete", text="演示旧规则：旧配送方式。")
    chunks = load_policy_chunks(settings.knowledge_dir)
    first = store.build(chunks)
    assert store.build(chunks) == first
    old_ids = {c.chunk_id for c in chunks}
    extra.unlink()
    write_policy(settings.knowledge_dir, text="演示新规则：请先联系客服确认退货。")
    current = load_policy_chunks(settings.knowledge_dir)
    rebuilt = store.build(current)
    assert rebuilt["index_fingerprint"] != first["index_fingerprint"]
    hits = store.query("退货？", top_k=4)
    assert {h.chunk_id for h in hits} == {c.chunk_id for c in current}
    assert not old_ids.intersection(h.chunk_id for h in hits)
    assert all("旧" not in h.text for h in hits)


def test_failed_build_keeps_previous_manifest_and_complete_collection(tmp_path):
    settings, embedding, store = setup_store(tmp_path)
    chunks = load_policy_chunks(settings.knowledge_dir)
    previous = store.build(chunks)
    write_policy(settings.knowledge_dir, text="演示规则更新：请联系原客服入口。")
    embedding.vector = [float("nan"), 0.0, 1.0]
    with pytest.raises(ShopError):
        store.build(load_policy_chunks(settings.knowledge_dir))
    assert json.loads((settings.chroma_dir / "manifest.json").read_text(encoding="utf-8")) == previous
    # Revert corpus and query through the original published collection.
    write_policy(settings.knowledge_dir)
    embedding.vector = None
    assert store.query("退货？", top_k=4)[0].chunk_id == chunks[0].chunk_id


def test_manifest_stat_permission_error_is_a_sanitized_503(tmp_path, monkeypatch):
    settings, embedding, store = setup_store(tmp_path)
    original_stat = Path.stat

    def denied_stat(path, *args, **kwargs):
        if path == store.manifest_path:
            raise PermissionError("private-stat-input " + str(tmp_path))
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", denied_stat)
        with pytest.raises(ShopError) as caught:
            store.query("private-question", top_k=4)
    assert caught.value.code == "rag_index_corrupt" and caught.value.http_status == 503
    assert "build_knowledge" in str(caught.value)
    assert "private" not in str(caught.value) and str(tmp_path) not in str(caught.value)
    assert embedding.calls == []


@pytest.mark.parametrize("outcome", ["published", "replace_failure"])
@pytest.mark.parametrize("cleanup_operation", ["stat", "unlink"])
def test_cleanup_io_errors_preserve_build_outcome_and_collection_cleanup(tmp_path, monkeypatch, outcome, cleanup_operation):
    settings, embedding, store = setup_store(tmp_path)
    previous = store.build(load_policy_chunks(settings.knowledge_dir))
    write_policy(settings.knowledge_dir, text="演示更新规则：先联系已有客服入口确认。")
    chunks = load_policy_chunks(settings.knowledge_dir)
    store_module = rag_module("store")
    client = store._client()
    original_delete = client.delete_collection
    original_replace = store_module.os.replace
    original_stat, original_exists, original_unlink = Path.stat, Path.exists, Path.unlink
    deleted = []

    def temporary(path):
        return path.parent == settings.chroma_dir and path.name.startswith(".manifest-") and path.suffix == ".tmp"

    def denied_stat(path, *args, **kwargs):
        if temporary(path):
            raise PermissionError("private-cleanup-stat " + str(tmp_path))
        return original_stat(path, *args, **kwargs)

    def denied_unlink(path, *args, **kwargs):
        if temporary(path):
            raise PermissionError("private-cleanup-unlink " + str(tmp_path))
        return original_unlink(path, *args, **kwargs)

    def existing_temporary(path):
        # Exercise an unlink access failure after successful publication even
        # if the normal replacement has removed the temporary directory entry.
        return True if temporary(path) else original_exists(path)

    def failed_replace(source, destination):
        if Path(destination) == store.manifest_path:
            raise OSError("private-replace-failure " + str(tmp_path))
        return original_replace(source, destination)

    def record_delete(name):
        deleted.append(name)
        return original_delete(name)

    with monkeypatch.context() as patch:
        patch.setattr(client, "delete_collection", record_delete)
        if cleanup_operation == "stat":
            patch.setattr(Path, "stat", denied_stat)
        else:
            patch.setattr(Path, "exists", existing_temporary)
            patch.setattr(Path, "unlink", denied_unlink)
        if outcome == "replace_failure":
            patch.setattr(store_module.os, "replace", failed_replace)
            with pytest.raises(ShopError) as caught:
                store.build(chunks)
            assert caught.value.code == "rag_unavailable" and caught.value.http_status == 503
            assert "private" not in str(caught.value) and str(tmp_path) not in str(caught.value)
        else:
            result = store.build(chunks)

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    collections = {collection.name for collection in client.list_collections()}
    if outcome == "replace_failure":
        assert manifest == previous
        assert len(deleted) == 1 and deleted[0] != previous["collection"]
        assert collections == {previous["collection"]}
        write_policy(settings.knowledge_dir)
        assert store.query("退货？", top_k=4)[0].document_id == "returns"
    else:
        assert result == manifest and result["index_fingerprint"] != previous["index_fingerprint"]
        assert deleted == [] and collections == {previous["collection"], result["collection"]}
        assert store.query("退货？", top_k=4)[0].text == chunks[0].text


@pytest.mark.parametrize("stage", ["after_collection_write", "replace"])
def test_staged_build_failure_preserves_old_index_and_removes_unpublished_artifacts(tmp_path, monkeypatch, stage):
    settings, embedding, store = setup_store(tmp_path)
    previous = store.build(load_policy_chunks(settings.knowledge_dir))
    write_policy(settings.knowledge_dir, text="演示新规则：退货前先确认寄回地址。")
    chunks = load_policy_chunks(settings.knowledge_dir)
    store_module = rag_module("store")
    from chromadb.api.models.Collection import Collection
    original_add = Collection.add
    original_replace = store_module.os.replace
    staged = []

    def fail_after_write(collection, *args, **kwargs):
        original_add(collection, *args, **kwargs)
        staged.append(collection.count())
        raise OSError("private-staged-write " + str(tmp_path))

    def fail_replace(source, destination):
        if Path(destination) == store.manifest_path:
            assert Path(source).is_file()
            staged.append(json.loads(Path(source).read_text(encoding="utf-8"))["chunk_count"])
            raise OSError("private-staged-publish " + str(tmp_path))
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        if stage == "after_collection_write":
            patch.setattr(Collection, "add", fail_after_write)
        else:
            patch.setattr(store_module.os, "replace", fail_replace)
        with pytest.raises(ShopError) as caught:
            store.build(chunks)
    assert caught.value.code == "rag_unavailable" and "private" not in str(caught.value)
    assert staged and staged[0] > 0, "failure must occur after the new collection has real persisted content"
    assert json.loads(store.manifest_path.read_text(encoding="utf-8")) == previous
    assert {collection.name for collection in store._client().list_collections()} == {previous["collection"]}
    assert list(settings.chroma_dir.glob(".manifest-*.tmp")) == []
    write_policy(settings.knowledge_dir)
    assert store.query("退货？", top_k=4)[0].document_id == "returns"


@pytest.mark.parametrize("change", ["model", "splitter", "corpus"])
def test_query_rejects_incompatible_model_splitter_or_corpus_before_embedding(tmp_path, change):
    settings, embedding, store = setup_store(tmp_path)
    store.build(load_policy_chunks(settings.knowledge_dir))
    other = DeterministicEmbeddings(identity="other" if change == "model" else "test-v1")
    if change == "splitter":
        settings = settings.model_copy(update={"chunk_size": 400})
    elif change == "corpus":
        write_policy(settings.knowledge_dir, text="演示规则更新了，请重建索引。")
    reopened = rag_module("store").PolicyStore(settings, other)
    with pytest.raises(ShopError) as caught:
        reopened.query("包含敏感输入的退货问题", top_k=4)
    assert caught.value.code == "rag_index_incompatible"
    assert other.calls == []
    assert "敏感输入" not in str(caught.value)


@pytest.mark.parametrize("defect", ["missing", "json", "collection", "count", "document"])
def test_missing_or_corrupt_index_has_actionable_sanitized_error(tmp_path, defect):
    settings, embedding, store = setup_store(tmp_path)
    if defect != "missing":
        manifest = store.build(load_policy_chunks(settings.knowledge_dir))
        manifest_path = settings.chroma_dir / "manifest.json"
        if defect == "json":
            manifest_path.write_text("private-error-not-json", encoding="utf-8")
        else:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False))
            if defect == "collection":
                client.delete_collection(manifest["collection"])
            else:
                collection = client.get_collection(manifest["collection"], embedding_function=None)
                if defect == "count":
                    collection.delete(ids=[load_policy_chunks(settings.knowledge_dir)[0].chunk_id])
                else:
                    # Explicit-vector collections require vectors on document
                    # updates; keep the vector unchanged while corrupting text.
                    collection.update(ids=[load_policy_chunks(settings.knowledge_dir)[0].chunk_id],
                                      documents=["private-corrupt-text"], embeddings=[[1.0, 0.0, 0.0]])
    with pytest.raises(ShopError) as caught:
        store.query("private-input", top_k=4)
    assert caught.value.code == ("rag_not_built" if defect == "missing" else "rag_index_corrupt")
    assert "build_knowledge" in str(caught.value)
    assert "private" not in str(caught.value) and str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("vector", [[1.0, 0.0], [float("nan"), 0.0, 1.0], [float("inf"), 0.0, 1.0], [0.0, 0.0, 0.0]])
def test_build_rejects_wrong_dimensions_nonfinite_and_zero_vectors(tmp_path, vector):
    settings, embedding, store = setup_store(tmp_path, embedding=DeterministicEmbeddings(vector=vector))
    with pytest.raises(ShopError) as caught:
        store.build(load_policy_chunks(settings.knowledge_dir))
    assert caught.value.code == "rag_unavailable"
    assert not (settings.chroma_dir / "manifest.json").exists()


@pytest.mark.parametrize("vector", [[1.0], [float("nan"), 0.0, 1.0], [float("inf"), 1.0, 0.0]])
def test_query_rejects_invalid_vectors(tmp_path, vector):
    settings, embedding, store = setup_store(tmp_path)
    store.build(load_policy_chunks(settings.knowledge_dir))
    embedding.vector = vector
    with pytest.raises(ShopError) as caught:
        store.query("退货？", top_k=4)
    assert caught.value.code == "rag_unavailable"


def test_no_evidence_answer_does_not_invent_a_policy(tmp_path):
    settings, embedding, store = setup_store(tmp_path)
    store.build(load_policy_chunks(settings.knowledge_dir))
    service = rag_module("service").PolicyService(settings, embeddings=embedding)
    result = service.answer("商城是否支持宠物医疗保险理赔？")
    assert result.status == "no_evidence" and result.sources == []
    assert "未找到" in result.answer and "客服" in result.answer
    assert "演示" in result.notice


def test_answer_only_quotes_returned_above_threshold_sources(tmp_path):
    settings, embedding, store = setup_store(tmp_path)
    store.build(load_policy_chunks(settings.knowledge_dir))
    result = rag_module("service").PolicyService(settings, embeddings=embedding).answer("  退货？  ")
    assert result.status == "answered" and result.question == "退货？"
    assert all(s.score >= settings.min_score and s.text in result.answer for s in result.sources)
    assert "演示" in result.answer and "演示" in result.notice


def test_rag_settings_defaults_paths_and_validation_are_independent_of_shop_settings():
    settings = rag_module("config").RagSettings(_env_file=None)
    assert settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert settings.min_score == 0.53 and settings.top_k == 4
    assert settings.chunk_size == 500 and settings.chunk_overlap == 50
    assert settings.model_cache_dir.is_relative_to(Path(__file__).resolve().parents[1])
    from pydantic import ValidationError
    for values in ({"min_score": float("nan")}, {"chunk_size": 601}, {"chunk_overlap": 500}, {"top_k": 0}):
        with pytest.raises(ValidationError):
            rag_module("config").RagSettings(_env_file=None, **values)


def test_local_embedding_fingerprint_has_real_weights_and_dependency_versions():
    config = rag_module("config")
    embeddings = rag_module("embeddings")
    model = embeddings.LocalEmbeddings(config.RagSettings(_env_file=None), allow_download=False)
    assert model.dimension == 512
    fingerprint = model.fingerprint()
    assert fingerprint["weights_sha256"] == "1294ea4b6331115a353d81f96b85e8c8d7fdcc284453d5b2fab5b016230aad38"
    assert {"fastembed", "onnxruntime", "chromadb", "langchain-text-splitters"} <= set(fingerprint["versions"])
    assert model._model is None, "fingerprinting must not load ONNX or download a model"
