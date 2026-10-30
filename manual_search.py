"""事前登録されたPDFマニュアルを検索するための処理。"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ManualChunk:
    filename: str
    page: int
    text: str


@dataclass(frozen=True)
class ManualHit:
    chunk: ManualChunk
    score: float


def split_into_chunks(text: str, size: int = 450, overlap: int = 80) -> list[str]:
    """ページ単位の参照情報を残しながら、長い文章を検索用に分割する。"""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
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
                for text in split_into_chunks(page.extract_text() or ""):
                    found_text = True
                    chunks.append(ManualChunk(path.name, page_number, text))
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
    vectors = vectorizer.fit_transform([chunk.text for chunk in chunks])
    query_vector = vectorizer.transform([query.strip()])
    scores = cosine_similarity(query_vector, vectors)[0]

    ordered = sorted(range(len(chunks)), key=lambda index: -scores[index])
    return [
        ManualHit(chunks[index], float(scores[index]))
        for index in ordered[:limit]
        if scores[index] >= threshold
    ]
