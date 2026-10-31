"""事前登録されたPDFマニュアルを検索するための処理。"""

from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ManualChunk:
    filename: str
    page: int
    topic: str
    heading: str
    text: str


@dataclass(frozen=True)
class ManualHit:
    chunk: ManualChunk
    score: float


def split_into_chunks(text: str, size: int = 650, overlap: int = 80) -> list[str]:
    """一般的なPDFでも改行を保持して検索用に分割する。"""
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            last_newline = normalized.rfind("\n", start + size // 2, end)
            if last_newline > start:
                end = last_newline
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_page_sections(text: str, filename: str, page: int) -> list[ManualChunk]:
    """番号付き見出しごとに抽出し、PDF内の手順と改行を維持する。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [
        line for line in lines
        if line != "IT"
        and not line.startswith("社内ITサポート｜デモ用マニュアル")
        and not line.startswith("本資料はポートフォリオ用")
        and not re.fullmatch(r"\d+\s*/\s*\d+", line)
    ]
    if not lines:
        return []

    topic = lines[0]
    headings = [i for i, line in enumerate(lines) if re.match(r"^\d+[.．]\s+", line)]
    if not headings:
        return [
            ManualChunk(filename, page, topic, "関連する記載", block)
            for block in split_into_chunks("\n".join(lines))
        ]

    notice_markers = ("セキュリティ上の注意", "他の利用者への配慮")
    notice_indexes = [i for i, line in enumerate(lines) if line in notice_markers]
    notice_start = notice_indexes[0] if notice_indexes else len(lines)

    chunks = []
    for index, start in enumerate(headings):
        end = headings[index + 1] if index + 1 < len(headings) else notice_start
        body = "\n".join(lines[start + 1:end]).strip()
        if body:
            chunks.append(ManualChunk(filename, page, topic, lines[start], body))
    if notice_indexes:
        notice_text = "\n".join(lines[notice_start + 1:]).strip()
        if notice_text:
            chunks.append(ManualChunk(filename, page, topic, lines[notice_start], notice_text))
    return chunks


def load_manuals(directory: Path) -> tuple[list[ManualChunk], list[str]]:
    """manuals/ 以下のPDFを読み、文字情報がないページは除外する。"""
    chunks: list[ManualChunk] = []
    warnings: list[str] = []

    for path in sorted(directory.glob("*.pdf")) if directory.exists() else []:
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                warnings.append(f"{path.name}: パスワード付きPDFは読み込めません。")
                continue
            found_text = False
            for page_number, page in enumerate(reader.pages, start=1):
                for chunk in extract_page_sections(
                    page.extract_text() or "", path.name, page_number
                ):
                    found_text = True
                    chunks.append(chunk)
            if not found_text:
                warnings.append(f"{path.name}: 抽出できる文字がありません（画像PDFは未対応）。")
        except Exception:
            warnings.append(f"{path.name}: PDFを読み込めませんでした。")

    return chunks, warnings


def search_manuals(
    query: str,
    chunks: list[ManualChunk],
    *,
    limit: int = 3,
    threshold: float = 0.12,
) -> list[ManualHit]:
    """文字2-3gramのTF-IDFで類似するマニュアル本文を返す。"""
    if not query.strip() or not chunks:
        return []

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    vectors = vectorizer.fit_transform(
        [f"{chunk.topic} {chunk.heading} {chunk.text}" for chunk in chunks]
    )
    query_vector = vectorizer.transform([query.strip()])
    scores = cosine_similarity(query_vector, vectors)[0]

    ordered = sorted(range(len(chunks)), key=lambda index: -scores[index])
    return [
        ManualHit(chunks[index], float(scores[index]))
        for index in ordered[:limit]
        if scores[index] >= threshold
    ]
