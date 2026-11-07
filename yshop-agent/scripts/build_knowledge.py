"""Explicit model preparation and atomic demo-policy index build.

Run from yshop-agent: python -m scripts.build_knowledge
Only this CLI enables remote model preparation; service queries stay offline.
"""

import json

from pydantic import ValidationError

from app.errors import ShopError
from app.rag.config import RagSettings
from app.rag.documents import load_policy_chunks
from app.rag.embeddings import LocalEmbeddings
from app.rag.store import PolicyStore


def main() -> int:
    try:
        settings = RagSettings()
        chunks = load_policy_chunks(settings.knowledge_dir, chunk_size=settings.chunk_size,
                                    chunk_overlap=settings.chunk_overlap)
        model = LocalEmbeddings(settings, allow_download=True)
        manifest = PolicyStore(settings, model).build(chunks)
        print(json.dumps({"status": "built", "notice": "仅为商城演示规则，不构成真实商城承诺。",
                          "model": settings.embedding_model, "dimension": model.dimension,
                          "document_count": manifest["document_count"], "chunk_count": manifest["chunk_count"],
                          "index_fingerprint": manifest["index_fingerprint"]}, ensure_ascii=False))
        return 0
    except (ValidationError, ValueError):
        print(json.dumps({"error": {"code": "invalid_request", "message": "RAG 配置或演示文档无效，请检查项目配置和规则文档后重试。"}}, ensure_ascii=False))
    except ShopError as error:
        print(json.dumps({"error": {"code": error.code, "message": error.message}}, ensure_ascii=False))
    except Exception:
        error = ShopError("rag_unavailable")
        print(json.dumps({"error": {"code": error.code, "message": error.message}}, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
