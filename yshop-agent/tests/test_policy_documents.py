import json
import os
from pathlib import Path
import subprocess

import pytest


POLICIES = Path(__file__).resolve().parents[1] / "knowledge" / "policies"


def write_policy(directory, name="rule.md", body=None, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "document_id": "demo-rule",
        "title": "测试演示规则",
        "source": f"knowledge/policies/{name}",
        "version": "2026-10-24.v1",
        "kind": "demo",
    }
    metadata.update(overrides)
    content = body if body is not None else "# 测试演示规则\n\n演示规则，仅供测试。\n\n## 申请说明\n\n演示规则：申请需提供订单号。\n"
    path = directory / name
    path.write_text("---\n" + json.dumps(metadata, ensure_ascii=False) + "\n---\n" + content, encoding="utf-8")
    return path


def test_real_policy_chunks_have_traceable_demo_sources():
    from app.rag.documents import load_policy_chunks

    chunks = load_policy_chunks(POLICIES)
    assert len(list(POLICIES.glob("*.md"))) == 15
    assert len({c.document_id for c in chunks}) == 15
    assert all(c.kind == "demo" and c.section and c.version for c in chunks)
    assert all(c.source.startswith("knowledge/policies/") for c in chunks)
    assert all(0 < len(c.text) <= 600 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all("演示规则" in c.text for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for topic in ("配送", "发货", "运费", "验货", "退货", "换货", "退款", "发票", "优惠券", "规格", "库存", "客服"):
        assert topic in joined
    for chunk in chunks:
        assert chunk.text in (POLICIES.parent.parent / chunk.source).read_text(encoding="utf-8")
        assert chunk.metadata == {
            "document_id": chunk.document_id, "title": chunk.title,
            "section": chunk.section, "source": chunk.source,
            "version": chunk.version, "kind": "demo", "chunk_id": chunk.chunk_id,
        }
        assert "document_id" not in chunk.text


def test_sections_and_paragraphs_preserve_original_text(tmp_path):
    from app.rag.documents import load_policy_chunks

    body = "# 标题\n\n## 配送\n\n演示规则：配送第一段。\n\n第二段保留换行。\n\n## 退款\n\n演示规则：退款另行处理。\n"
    write_policy(tmp_path, body=body)
    chunks = load_policy_chunks(tmp_path)
    assert [c.section for c in chunks] == ["配送", "退款"]
    assert "第一段。\n\n第二段" in chunks[0].text
    assert "退款" not in chunks[0].text
    assert "配送" not in chunks[1].text
    assert all(c.text in body for c in chunks)


def test_nested_sections_keep_context_and_fenced_headings_are_data(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, body="## 退货\n\n演示规则：先申请。\n\n### 材料\n\n演示规则：提供照片。\n\n```text\n## 不是真章节\n保留原文\n```\n")
    chunks = load_policy_chunks(tmp_path)
    assert [c.section for c in chunks] == ["退货", "退货 / 材料"]
    assert "## 不是真章节\n保留原文" in chunks[1].text


def test_long_paragraph_keeps_overlap_and_respects_hard_limit(tmp_path):
    from app.rag.documents import load_policy_chunks

    paragraph = "".join(chr(0x4E00 + i) for i in range(1700))
    write_policy(tmp_path, body=f"## 长段落\n\n{paragraph}\n")
    chunks = load_policy_chunks(tmp_path)
    assert len(chunks) >= 4
    assert all(0 < len(c.text) <= 500 for c in chunks)
    assert all(c.section == "长段落" for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-50:] == current.text[:50]
    assert all(c.text in f"## 长段落\n\n{paragraph}" for c in chunks)


@pytest.mark.parametrize("paragraph", [
    "甲" * 250 + "。" + "乙" * 250 + "。",
    "".join(
        "".join(chr(0x4E00 + sentence * 100 + i) for i in range(100)) + "。"
        for sentence in range(17)
    ),
], ids=["two-sentence-minimal", "unique-sentences"])
def test_long_chinese_sentences_keep_actual_fifty_character_overlap(tmp_path, paragraph):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, body=f"## 配送详情\n\n{paragraph}\n")
    chunks = load_policy_chunks(tmp_path)
    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 500 for chunk in chunks)
    spans = [(paragraph.find(chunk.text), len(chunk.text)) for chunk in chunks]
    assert all(start >= 0 for start, _ in spans)
    assert spans[0][0] == 0
    assert spans[-1][0] + spans[-1][1] == len(paragraph)
    for (previous_start, previous_length), (start, _) in zip(spans, spans[1:]):
        assert previous_start + previous_length - start == 50
    assert chunks == load_policy_chunks(tmp_path)


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(600, 50), (120, 25), (120, 0)])
def test_punctuated_paragraph_respects_configured_character_overlap(tmp_path, chunk_size, chunk_overlap):
    from app.rag.documents import load_policy_chunks

    paragraph = "".join(
        "".join(chr(0x4E00 + sentence * 80 + i) for i in range(80)) + "！"
        for sentence in range(20)
    )
    write_policy(tmp_path, body=f"## 长段落\n\n{paragraph}\n")
    chunks = load_policy_chunks(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert all(0 < len(chunk.text) <= chunk_size <= 600 for chunk in chunks)
    spans = [(paragraph.find(chunk.text), len(chunk.text)) for chunk in chunks]
    assert all(start >= 0 for start, _ in spans)
    assert spans[0][0] == 0
    assert spans[-1][0] + spans[-1][1] == len(paragraph)
    for (previous_start, previous_length), (start, _) in zip(spans, spans[1:]):
        assert previous_start + previous_length - start == chunk_overlap


def test_internal_whitespace_at_chunk_edges_does_not_shorten_overlap(tmp_path):
    from app.rag.documents import load_policy_chunks

    paragraph = "".join(
        " " if i in {450, 499, 900, 949} else chr(0x4E00 + i)
        for i in range(1700)
    )
    write_policy(tmp_path, body=f"## 长段落\n\n{paragraph}\n")
    chunks = load_policy_chunks(tmp_path)
    spans = [(paragraph.find(chunk.text), len(chunk.text)) for chunk in chunks]
    assert all(start >= 0 for start, _ in spans)
    for (previous_start, previous_length), (start, _) in zip(spans, spans[1:]):
        assert previous_start + previous_length - start == 50


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(500, 50), (600, 50), (120, 25), (120, 0)])
def test_soft_newlines_in_one_markdown_paragraph_keep_exact_overlap(tmp_path, chunk_size, chunk_overlap):
    from app.rag.documents import load_policy_chunks

    paragraph = "\n".join(
        "".join(chr(0x4E00 + line * 100 + i) for i in range(100))
        for line in range(6)
    )
    assert len(paragraph) == 605 and "\n\n" not in paragraph
    write_policy(tmp_path, body=f"## 配送详情\n\n{paragraph}\n")
    chunks = load_policy_chunks(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1
    assert all(chunk.section == "配送详情" for chunk in chunks)
    assert all(0 < len(chunk.text) <= chunk_size <= 600 for chunk in chunks)
    spans = [(paragraph.find(chunk.text), len(chunk.text)) for chunk in chunks]
    assert all(start >= 0 for start, _ in spans)
    assert spans[0][0] == 0
    assert spans[-1][0] + spans[-1][1] == len(paragraph)
    for (previous_start, previous_length), (start, _) in zip(spans, spans[1:]):
        assert previous_start + previous_length - start == chunk_overlap
    assert chunks == load_policy_chunks(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_short_paragraphs_stay_whole_when_a_section_needs_splitting(tmp_path):
    from app.rag.documents import load_policy_chunks

    paragraphs = [letter * 280 for letter in "甲乙丙"]
    body = "\n\n".join(paragraphs)
    write_policy(tmp_path, body=f"## 段落边界\n\n{body}\n")
    chunks = load_policy_chunks(tmp_path)
    assert len(chunks) == 3
    assert all(paragraph in chunk.text for paragraph, chunk in zip(paragraphs, chunks))
    assert all(chunk.text in body for chunk in chunks)


def test_chunk_ids_are_stable_and_change_with_content(tmp_path):
    from app.rag.documents import load_policy_chunks

    path = write_policy(tmp_path)
    before = load_policy_chunks(tmp_path)
    assert before == load_policy_chunks(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("订单号", "订单号与照片"), encoding="utf-8")
    after = load_policy_chunks(tmp_path)
    assert before[0].chunk_id == after[0].chunk_id
    assert before[-1].chunk_id != after[-1].chunk_id


def test_identical_repeated_chunks_have_distinct_stable_ids(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, body="## 材料\n\n" + "甲" * 2000)
    chunks = load_policy_chunks(tmp_path, chunk_size=100, chunk_overlap=20)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks == load_policy_chunks(tmp_path, chunk_size=100, chunk_overlap=20)


def test_duplicate_document_ids_are_rejected(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, "a.md")
    write_policy(tmp_path, "b.md")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


def test_duplicate_sources_are_rejected(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, "a.md")
    write_policy(tmp_path, "b.md", document_id="demo-second", source="knowledge/policies/a.md")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


@pytest.mark.parametrize("field", ["document_id", "title", "source", "version", "kind"])
@pytest.mark.parametrize("value", [None, "", 42])
def test_missing_or_invalid_metadata_is_rejected(tmp_path, field, value):
    from app.rag.documents import load_policy_chunks

    path = write_policy(tmp_path, **{field: value})
    if value is None:
        lines = path.read_text(encoding="utf-8").split("\n", 3)
        metadata = json.loads(lines[1])
        del metadata[field]
        path.write_text("---\n" + json.dumps(metadata) + "\n---\n" + lines[3], encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


@pytest.mark.parametrize("body", ["", "  \n", "# 仅有标题\n", "没有二级章节的正文", "## \n文字", "## 申请\n \n", "## 申请\n### 材料\n"])
def test_invalid_or_empty_body_is_rejected(tmp_path, body):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, body=body)
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


@pytest.mark.parametrize("front_matter", ["title: 非 JSON", "[]", "{broken", "{\"document_id\": \"one\", \"document_id\": \"two\"}"])
def test_invalid_front_matter_is_rejected(tmp_path, front_matter):
    from app.rag.documents import load_policy_chunks

    (tmp_path / "rule.md").write_text(f"---\n{front_matter}\n---\n## 规则\n正文", encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


@pytest.mark.parametrize("source", ["knowledge/policies/../outside.md", "../outside.md", "knowledge/policies/wrong.md", "knowledge\\policies\\rule.md", "https://example.test/rule.md", "C:/private/rule.md"])
def test_source_must_match_actual_safe_relative_path(tmp_path, source):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, source=source)
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


def test_non_demo_documents_are_rejected(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, kind="official")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(601, 50), (0, 0), (500, -1), (50, 50), (True, 0), (500, 1.5)])
def test_invalid_splitter_configuration_is_rejected(tmp_path, chunk_size, chunk_overlap):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path)
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_missing_and_empty_directory_are_rejected(tmp_path):
    from app.rag.documents import load_policy_chunks

    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)


def test_symlink_file_cannot_escape_policy_directory(tmp_path):
    from app.rag.documents import load_policy_chunks

    rules = tmp_path / "rules"
    outside = write_policy(tmp_path / "outside")
    rules.mkdir()
    try:
        (rules / "rule.md").symlink_to(outside)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows does not grant symbolic-link creation privilege")
        raise
    with pytest.raises(ValueError):
        load_policy_chunks(rules)


def test_symlink_directory_cannot_escape_policy_directory(tmp_path):
    from app.rag.documents import load_policy_chunks

    rules = tmp_path / "rules"
    write_policy(rules)
    outside = tmp_path / "outside"
    write_policy(outside, document_id="outside-rule")
    try:
        (rules / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows does not grant symbolic-link creation privilege")
        raise
    with pytest.raises(ValueError):
        load_policy_chunks(rules)


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-junction boundary")
@pytest.mark.parametrize("load_link_itself", [False, True])
def test_windows_junction_cannot_escape_policy_directory(tmp_path, load_link_itself):
    from app.rag.documents import load_policy_chunks

    rules = tmp_path / "rules"
    write_policy(rules)
    outside = tmp_path / "outside"
    write_policy(outside, document_id="outside-rule")
    link = rules / "linked"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, "Could not create project-local test directory junction"
    assert link.is_junction()
    with pytest.raises(ValueError):
        load_policy_chunks(link if load_link_itself else rules)


def test_meaningful_section_named_demo_notice_is_valid(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, body="## 演示说明\n\n演示规则：仅供测试，不代表实际承诺。\n")
    chunks = load_policy_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].section == "演示说明"


def test_introduction_does_not_make_empty_rule_section_valid(tmp_path):
    from app.rag.documents import load_policy_chunks

    write_policy(tmp_path, body="演示规则，仅供测试。\n\n## 配送\n\n")
    with pytest.raises(ValueError):
        load_policy_chunks(tmp_path)
