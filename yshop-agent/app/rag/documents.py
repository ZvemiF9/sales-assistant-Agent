"""Validate demonstration Markdown policies and split their original text.

This module loads only the supplied policy directory. It does not contact the
shop, initialize embedding models, or execute instructions in document text.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


_FIELDS = ("document_id", "title", "source", "version", "kind")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", re.DOTALL)
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True, slots=True)
class PolicyChunk:
    document_id: str
    title: str
    section: str
    source: str
    version: str
    kind: str
    chunk_id: str
    text: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "version": self.version,
            "kind": self.kind,
            "chunk_id": self.chunk_id,
        }


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("规则 JSON 元数据包含重复字段，请修正文档。")
        result[key] = value
    return result


def _policy_paths(root: Path) -> list[Path]:
    # Windows junctions are not necessarily reported as symbolic links.
    if any(p.is_symlink() or p.is_junction() for p in (root, *root.parents)):
        raise ValueError("规则目录不能使用符号链接或目录联接，请使用项目内真实目录。")
    if not root.is_dir():
        raise ValueError("规则目录不存在，请先创建 knowledge/policies 并添加演示规则。")

    resolved_root = root.resolve(strict=True)
    paths = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in sorted(directory.iterdir()):
            if entry.is_symlink() or entry.is_junction():
                raise ValueError("规则目录中存在符号链接或目录联接，请移除链接后重试。")
            if not entry.resolve(strict=True).is_relative_to(resolved_root):
                raise ValueError("规则路径超出规则目录，请检查文档位置。")
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file() and entry.suffix.lower() == ".md":
                paths.append(entry)
    if not paths:
        raise ValueError("规则目录中没有 Markdown 文档，请添加演示规则后重试。")
    return sorted(paths)


def _load_document(path: Path, root: Path) -> tuple[dict[str, str], str]:
    content = path.read_text(encoding="utf-8-sig")
    match = _FRONT_MATTER.fullmatch(content)
    if match is None:
        raise ValueError("规则文档缺少有效的 JSON front matter，请检查两行 --- 分隔符。")
    try:
        metadata = json.loads(match[1], object_pairs_hook=_unique_json_object)
    except (ValueError, TypeError):
        raise ValueError("规则 JSON 元数据无效或存在重复字段，请修正文档。") from None
    if not isinstance(metadata, dict) or set(metadata) != set(_FIELDS):
        raise ValueError("规则元数据必须包含 document_id、title、source、version 和 kind。")
    if any(not isinstance(metadata[key], str) or not metadata[key].strip() for key in _FIELDS):
        raise ValueError("规则元数据字段必须为非空字符串，请修正文档。")
    if any(metadata[key] != metadata[key].strip() for key in _FIELDS):
        raise ValueError("规则元数据字段不能包含首尾空白，请修正文档。")
    expected_source = "knowledge/policies/" + path.relative_to(root).as_posix()
    if metadata["source"] != expected_source:
        raise ValueError("规则来源必须与 knowledge/policies 下的实际文档相符，请修正 source。")
    if metadata["kind"] != "demo":
        raise ValueError("当前知识库仅允许 kind 为 demo 的演示规则。")
    return metadata, match[2]


def _sections(body: str) -> list[tuple[str, str]]:
    sections = []
    headings: dict[int, str] = {}
    section = "演示说明"
    lines: list[str] = []
    has_level_two = False
    has_rule_content = False
    fence: tuple[str, int] | None = None

    def finish_section() -> None:
        nonlocal has_rule_content
        text = "".join(lines).strip()
        if text:
            sections.append((section, text))
            has_rule_content |= has_level_two

    for line in body.splitlines(keepends=True):
        fence_match = _FENCE.match(line.rstrip("\r\n"))
        if fence is not None:
            lines.append(line)
            if fence_match and fence_match[1][0] == fence[0] and len(fence_match[1]) >= fence[1] and not fence_match[2].strip():
                fence = None
            continue
        if fence_match:
            fence = (fence_match[1][0], len(fence_match[1]))
            lines.append(line)
            continue
        heading_match = _HEADING.match(line.rstrip("\r\n"))
        if heading_match is None:
            lines.append(line)
            continue
        level = len(heading_match[1])
        title = re.sub(r"[ \t]+#+[ \t]*$", "", heading_match[2] or "").strip()
        if not title:
            raise ValueError("规则章节标题不能为空，请添加有意义的标题。")
        if level == 1:
            # The document title lives in metadata and is not vector text.
            if has_level_two:
                raise ValueError("规则正文不能重复使用一级标题，请按二级章节组织规则。")
            continue
        finish_section()
        lines = []
        has_level_two |= level == 2
        headings = {depth: value for depth, value in headings.items() if depth < level}
        headings[level] = title
        section = " / ".join(headings[depth] for depth in sorted(headings))
    finish_section()
    if not has_level_two or not sections:
        raise ValueError("规则正文必须包含至少一个有内容的二级章节，请补充演示规则。")
    # An introduction alone must not make an empty rule section valid.
    if not has_rule_content:
        raise ValueError("规则章节正文为空，请补充演示规则。")
    return sections


def load_policy_chunks(
    knowledge_dir: Path, *, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[PolicyChunk]:
    """Return deterministic chunks with traceable, validated demo metadata.

    Split each Markdown section independently. Paragraph boundaries are
    preferred; long paragraphs fall back to character splitting with overlap.
    Chunk identity hashes source metadata, section, ordinal and exact text.
    Invalid input raises a ValueError that never reflects raw document content.
    """
    if type(chunk_size) is not int or not 0 < chunk_size <= 600:
        raise ValueError("chunk_size 必须为 1 至 600 的整数。")
    if type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必须为非负整数并小于 chunk_size。")
    root = Path(knowledge_dir).absolute()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Sentence- or line-sized pieces can exceed chunk_overlap and cause
        # the merger to drop all context. Only blank lines divide paragraphs;
        # punctuation and Markdown soft newlines remain overlapping characters.
        separators=["\n\n", ""],
        keep_separator=True,
        strip_whitespace=False,
    )
    chunks = []
    document_ids: set[str] = set()
    sources: set[str] = set()
    try:
        for path in _policy_paths(root):
            metadata, body = _load_document(path, root)
            if metadata["document_id"] in document_ids or metadata["source"] in sources:
                raise ValueError("规则文档 ID 或来源重复，请为每篇规则使用唯一元数据。")
            document_ids.add(metadata["document_id"])
            sources.add(metadata["source"])
            for section_index, (section, text) in enumerate(_sections(body)):
                for chunk_index, part in enumerate(splitter.split_text(text)):
                    # Section edges are already stripped. Keep internal edge
                    # whitespace, which is part of the exact overlap and source.
                    if not part.strip():
                        continue
                    identity = json.dumps(
                        [metadata, section, section_index, chunk_index, part],
                        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                    )
                    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    chunks.append(PolicyChunk(**metadata, section=section, chunk_id=chunk_id, text=part))
    except (OSError, UnicodeError):
        raise ValueError("无法读取规则目录或 UTF-8 文档，请检查项目目录权限和文件编码。") from None
    return chunks
