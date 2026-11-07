"""Explicit-vector Chroma persistence with verified, generation-bound manifests."""

import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
import os
import re
import uuid

from app.errors import ShopError
from app.rag.config import RagSettings
from app.rag.documents import PolicyChunk, load_policy_chunks
from app.rag.embeddings import Embeddings, configure_model_environment, _quiet_logger
from app.rag.schemas import PolicySource


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _content(chunks: list[PolicyChunk]) -> list[dict]:
    return sorted([dict(chunk.metadata, text=chunk.text) for chunk in chunks], key=lambda row: row["chunk_id"])


def _vector(vector, dimension: int) -> list[float]:
    try:
        values = list(vector)
        if len(values) != dimension or any(isinstance(x, bool) or not isinstance(x, Real) for x in values):
            raise ValueError
        result = [float(x) for x in values]
        if not all(math.isfinite(x) for x in result) or not any(x != 0 for x in result):
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise ShopError("rag_unavailable") from None


class PolicyStore:
    def __init__(self, settings: RagSettings, embeddings: Embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self.manifest_path = settings.chroma_dir / "manifest.json"
        self._client_instance = None

    def _client(self):
        if self._client_instance is None:
            configure_model_environment(self.settings.model_cache_dir)
            _quiet_logger("chromadb")
            from chromadb import PersistentClient
            from chromadb.config import Settings as ChromaSettings
            self._client_instance = PersistentClient(
                path=str(self.settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client_instance

    def _identity(self, chunks: list[PolicyChunk]) -> dict:
        content_hash = _digest(_content(chunks))
        identity = {
            "schema_version": 1,
            "model_fingerprint": self.embeddings.fingerprint(),
            "corpus_fingerprint": content_hash,
            "splitter_fingerprint": _digest({
                "algorithm": "markdown-sections-blank-paragraphs-exact-overlap-v1",
                "chunk_size": self.settings.chunk_size, "chunk_overlap": self.settings.chunk_overlap,
                "langchain-text-splitters": version("langchain-text-splitters"),
            }),
            "chunk_count": len(chunks),
            "document_count": len({chunk.document_id for chunk in chunks}),
        }
        return dict(identity, index_fingerprint=_digest(identity))

    def _manifest(self) -> dict:
        try:
            if not self.manifest_path.exists():
                raise ShopError("rag_not_built")
            if self.manifest_path.is_symlink():
                raise ValueError
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            fields = {"schema_version", "model_fingerprint", "corpus_fingerprint", "splitter_fingerprint",
                      "chunk_count", "document_count", "index_fingerprint", "collection"}
            if not isinstance(manifest, dict) or set(manifest) != fields:
                raise ValueError
            if manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int:
                raise ValueError
            for name in ("chunk_count", "document_count"):
                if type(manifest[name]) is not int or manifest[name] < 1:
                    raise ValueError
            if not isinstance(manifest["model_fingerprint"], dict):
                raise ValueError
            for name in ("corpus_fingerprint", "splitter_fingerprint", "index_fingerprint"):
                if not isinstance(manifest[name], str) or re.fullmatch(r"[a-f0-9]{64}", manifest[name]) is None:
                    raise ValueError
            collection = manifest["collection"]
            if not isinstance(collection, str) or re.fullmatch(r"policies-[a-f0-9]{16}-[a-f0-9]{32}", collection) is None:
                raise ValueError
            identity = {key: value for key, value in manifest.items() if key not in {"index_fingerprint", "collection"}}
            if _digest(identity) != manifest["index_fingerprint"]:
                raise ValueError
            return manifest
        except (OSError, UnicodeError, ValueError, TypeError):
            raise ShopError("rag_index_corrupt") from None

    def _collection(self, manifest: dict):
        try:
            collection = self._client().get_collection(manifest["collection"], embedding_function=None)
            if collection.count() != manifest["chunk_count"]:
                raise ValueError
            # Parsing .configuration reconstructs legacy embedding functions
            # and warns in Chroma 1.5.9 even for explicit-vector collections.
            if collection.configuration_json["hnsw"]["space"] != "cosine":
                raise ValueError
            data = collection.get(include=["documents", "metadatas", "embeddings"])
            records = []
            if len(data["ids"]) != manifest["chunk_count"]:
                raise ValueError
            for chunk_id, text, metadata, vector in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"], strict=True):
                source = PolicySource(**metadata, text=text, score=1.0)
                if source.chunk_id != chunk_id or not source.text.strip() or len(source.text) > 600:
                    raise ValueError
                _vector(vector, self.embeddings.dimension)
                records.append(source.model_dump(exclude={"score"}))
            if _digest(sorted(records, key=lambda row: row["chunk_id"])) != manifest["corpus_fingerprint"]:
                raise ValueError
            return collection
        except Exception:
            raise ShopError("rag_index_corrupt") from None

    def _current_chunks(self) -> list[PolicyChunk]:
        try:
            return load_policy_chunks(self.settings.knowledge_dir,
                                      chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap)
        except ValueError:
            raise ShopError("rag_index_incompatible") from None

    def build(self, chunks: list[PolicyChunk]) -> dict:
        """Publish only complete generations; an identical valid build is reused.

        Superseded collections remain available to readers holding the previous
        manifest. New queries always select the single published collection.
        """
        collection_name = None
        temporary = None
        published = False
        try:
            if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
                raise ValueError
            for chunk in chunks:
                source = PolicySource(**chunk.metadata, text=chunk.text, score=1.0)
                if not source.text.strip() or len(source.text) > 600:
                    raise ValueError
            identity = self._identity(chunks)
            if self.manifest_path.exists():
                try:
                    previous = self._manifest()
                    if all(previous[key] == value for key, value in identity.items()):
                        self._collection(previous)
                        return previous
                except ShopError:
                    # A corrupt generation can be repaired by this explicit CLI.
                    pass
            vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise ValueError
            vectors = [_vector(vector, self.embeddings.dimension) for vector in vectors]
            collection_name = f"policies-{identity['index_fingerprint'][:16]}-{uuid.uuid4().hex}"
            collection = self._client().create_collection(
                name=collection_name, embedding_function=None,
                configuration={"hnsw": {"space": "cosine", "num_threads": 2}},
            )
            # Batch writes respect Chroma's maximum batch size and avoid a single
            # huge request when a demonstration document is expanded.
            batch_size = min(256, self._client().get_max_batch_size())
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset:offset + batch_size]
                collection.add(ids=[chunk.chunk_id for chunk in batch],
                               documents=[chunk.text for chunk in batch],
                               metadatas=[chunk.metadata for chunk in batch],
                               embeddings=vectors[offset:offset + batch_size])
            manifest = dict(identity, collection=collection_name)
            self._collection(manifest)
            temporary = self.manifest_path.with_name(f".manifest-{uuid.uuid4().hex}.tmp")
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
            published = True
            return manifest
        except ShopError:
            raise
        except Exception:
            raise ShopError("rag_unavailable") from None
        finally:
            # Best-effort cleanup only of this unpublished attempt. Never delete
            # any published generation, which another reader may still be using.
            if temporary is not None:
                try:
                    if temporary.exists():
                        temporary.unlink()
                except OSError:
                    pass
            if collection_name is not None and not published:
                try:
                    self._client().delete_collection(collection_name)
                except Exception:
                    pass

    def query(self, question: str, *, top_k: int = 4) -> list[PolicySource]:
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ShopError("invalid_request")
        manifest = self._manifest()
        expected = self._identity(self._current_chunks())
        if any(manifest[key] != value for key, value in expected.items()):
            raise ShopError("rag_index_incompatible")
        collection = self._collection(manifest)
        vector = _vector(self.embeddings.embed_query(question), self.embeddings.dimension)
        try:
            result = collection.query(query_embeddings=[vector], n_results=min(top_k, manifest["chunk_count"]),
                                      include=["documents", "metadatas", "distances"])
            hits = []
            for chunk_id, text, metadata, distance in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True):
                if not isinstance(distance, Real) or not math.isfinite(distance) or not -1e-5 <= distance <= 2.00001:
                    raise ValueError
                score = max(-1.0, min(1.0, 1.0 - float(distance)))
                if score >= self.settings.min_score:
                    source = PolicySource(**metadata, text=text, score=score)
                    if source.chunk_id != chunk_id:
                        raise ValueError
                    hits.append(source)
            return sorted(hits, key=lambda source: (-source.score, source.chunk_id))
        except Exception:
            raise ShopError("rag_index_corrupt") from None
