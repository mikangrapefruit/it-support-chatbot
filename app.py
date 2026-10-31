"""社内ITサポート FAQ検索アプリ"""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from manual_search import ManualHit, load_manuals, search_manuals


st.set_page_config(
    page_title="社内ITサポート | FAQ検索",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --page-bg: #f5f7f9;
        --surface: #ffffff;
        --heading: #182d41;
        --text: #344559;
        --muted: #68798a;
        --brand: #27597c;
        --border: #dfe6ed;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN",
                     "Yu Gothic", "Meiryo", sans-serif;
    }

    [data-testid="stAppViewContainer"] { background-color: var(--page-bg); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"], .block-container {
        max-width: 1000px;
        padding-top: 2.1rem;
        padding-bottom: 3.5rem;
    }

    .brandbar {
        display: flex; align-items: center; gap: 11px;
        padding-bottom: 18px; border-bottom: 1px solid var(--border);
        font-size: 13px; font-weight: 700; color: #3c5064;
        letter-spacing: .025em;
    }
    .brandmark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; border-radius: 6px;
        background: #213e59; color: white; font-size: 13px; font-weight: 800;
        letter-spacing: -.03em;
    }
    .page-intro { margin: 33px 0 27px; }
    .page-intro h1 {
        color: var(--heading); font-size: 31px; font-weight: 750;
        line-height: 1.4; margin: 0 0 9px; letter-spacing: -.045em;
    }
    .page-intro p {
        font-size: 14px; color: #596c7e; margin: 0; line-height: 1.95;
    }
    .intro-meta {
        display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
        margin-top: 17px; font-size: 12px; color: var(--muted);
    }
    .meta-pill {
        background: #e9eef3; color: #44586d;
        border-radius: 4px; padding: 4px 9px; font-weight: 600;
    }
    .meta-separator { color: #b7c3ce; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 28px; border-bottom: 1px solid #d8e1e9;
    }
    .stTabs [data-baseweb="tab"] {
        height: 51px; background: transparent; color: #657589;
        font-size: 14px; font-weight: 600; padding: 0 2px;
    }
    .stTabs [aria-selected="true"] {
        color: #1d4c6c !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #2d648a; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 27px; }

    .section-title {
        font-size: 18px; font-weight: 750; letter-spacing: -.025em;
        color: #20384d; margin: 0 0 9px;
    }
    .section-description {
        font-size: 13px; line-height: 1.85; color: var(--muted);
        margin: 0 0 20px;
    }

    [data-testid="stForm"] {
        background: var(--surface); border: 1px solid var(--border) !important;
        border-radius: 10px !important; padding: 22px 23px 17px !important;
    }
    [data-testid="stTextInput"] input {
        background: #fff; border-radius: 6px;
    }
    [data-testid="stFormSubmitButton"] button,
    div.stButton > button[kind="primary"] {
        background-color: var(--brand); border-color: var(--brand);
        border-radius: 6px; font-weight: 700;
    }
    [data-testid="stFormSubmitButton"] button:hover,
    div.stButton > button[kind="primary"]:hover {
        background-color: #174665; border-color: #174665;
    }
    .stButton button { border-radius: 6px; }

    .answer-card {
        margin-top: 22px; padding: 24px 27px 20px; background: #fff;
        border: 1px solid var(--border); border-radius: 10px;
        border-left: 4px solid #315e7f;
    }
    .answer-head {
        display: flex; align-items: center; gap: 12px;
        justify-content: space-between; flex-wrap: wrap; margin-bottom: 14px;
    }
    .answer-eyebrow {
        font-size: 12px; color: #617589; font-weight: 750;
        letter-spacing: .025em;
    }
    .answer-category {
        color: #355b76; background: #edf3f7; font-size: 12px;
        padding: 5px 10px; border-radius: 4px;
    }
    .answer-question {
        color: #1a3044; font-size: 19px; font-weight: 730;
        line-height: 1.6; margin: 0 0 17px;
    }
    .answer-rule { border: 0; border-top: 1px solid #e9eef2; margin: 0 0 17px; }
    .answer-label { color: #42576a; font-size: 12px; font-weight: 700; margin-bottom: 6px; }
    .answer-body { color: #283c50; font-size: 15px; line-height: 1.95; margin: 0 0 23px; }
    .answer-source {
        border-top: 1px solid #edf0f3; padding-top: 12px;
        font-size: 12px; color: #718093; line-height: 1.8;
    }
    .answer-source strong { color: #506477; font-weight: 600; }

    .empty-result {
        background: #fff; border: 1px solid var(--border);
        border-radius: 10px; padding: 21px 24px; margin-top: 22px;
    }
    .empty-result-title { font-weight: 700; font-size: 15px; color: #2b4053; }
    .empty-result-body { margin-top: 7px; font-size: 13px; color: #68798b; line-height: 1.85; }

    .helper-note {
        font-size: 12px; line-height: 1.8; color: #738293;
        margin-top: 13px;
    }
    .app-footer {
        border-top: 1px solid var(--border); padding-top: 18px;
        margin-top: 43px; color: #788694; font-size: 12px;
        line-height: 1.9;
    }

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"], .block-container { padding-top: 1.1rem; }
        .page-intro { margin-top: 26px; }
        .page-intro h1 { font-size: 26px; }
        .stTabs [data-baseweb="tab-list"] { gap: 15px; }
        .stTabs [data-baseweb="tab"] { font-size: 12px; }
        .answer-card { padding: 20px 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_faq() -> pd.DataFrame:
    """CSVを読み込む。元の列名・データ形式は変更しない。"""
    path = Path(__file__).resolve().parent / "faq.csv"
    data = pd.read_csv(path, dtype={"id": str}).fillna("")
    expected_columns = {"id", "category", "question", "answer", "source"}
    if not expected_columns.issubset(data.columns):
        raise ValueError("faq.csv に必要な列がありません。")
    return data


def clean(value: object) -> str:
    """表示前にHTML特殊文字をエスケープする。"""
    return escape(str(value), quote=True)


def display_answer(row: pd.Series, label: str = "検索結果") -> None:
    
    answer_text = clean(row["answer"]).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="answer-card">
          <div class="answer-head">
            <span class="answer-eyebrow">{clean(label)}</span>
            <span class="answer-category">{clean(row['category'])}</span>
          </div>
          <div class="answer-question">{clean(row['question'])}</div>
          <hr class="answer-rule">
          <div class="answer-label">対応方法</div>
          <p class="answer-body">{answer_text}</p>
          <div class="answer-source">参照元　<strong>{clean(row['source'])}</strong>（架空の資料）</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


FAQ_SEARCH_ALIASES = {
    "パソコンが起動しません": ("電源が入らない", "電源がつかない"),
    "パソコンの動作が遅いです": ("パソコンが重い", "動作がもたつく"),
    "メールを受信できません": ("メールが届かない", "受信メールが来ない"),
}


def search_questions(question: str, data: pd.DataFrame, limit: int = 3) -> list[int]:
    """質問文と言い換え表現から、関連するFAQを最大3件返す。"""
    search_texts = []
    owners = []
    for index, faq_question in enumerate(data["question"].astype(str)):
        for text in (faq_question, *FAQ_SEARCH_ALIASES.get(faq_question, ())):
            search_texts.append(text)
            owners.append(index)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    vectors = vectorizer.fit_transform(search_texts)
    question_vector = vectorizer.transform([question.strip()])
    text_scores = cosine_similarity(question_vector, vectors)[0]
    scores = [0.0] * len(data)
    for owner, score in zip(owners, text_scores):
        scores[owner] = max(scores[owner], float(score))
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    if not ranked or scores[ranked[0]] < 0.30:
        return []
    return [index for index in ranked if scores[index] >= 0.15][:limit]


MANUAL_DIRECTORY = Path(__file__).resolve().parent / "manuals"


def manual_file_signature() -> tuple[tuple[str, int, int], ...]:
    """PDFを追加・更新した時にキャッシュを更新する。"""
    if not MANUAL_DIRECTORY.exists():
        return ()
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(MANUAL_DIRECTORY.glob("*.pdf"))
    )


@st.cache_data(show_spinner=False)
def cached_manuals(signature: tuple[tuple[str, int, int], ...]):
    # signature はキャッシュの更新判定に利用する。
    return load_manuals(MANUAL_DIRECTORY)


def display_manual_hit(hit: ManualHit) -> None:
    """マニュアルの抜粋を、文書名・ページ番号とともに表示する。"""
    text = clean(hit.chunk.text)
    st.markdown(
        f"""
        <div class="answer-card">
          <div class="answer-head">
            <span class="answer-eyebrow">マニュアルの該当箇所</span>
            <span class="answer-category">PDF資料</span>
          </div>
          <div class="answer-question">{clean(hit.chunk.filename)}</div>
          <hr class="answer-rule">
          <div class="answer-label">該当する記載（抜粋）</div>
          <p class="answer-body">{text}</p>
          <div class="answer-source">参照元　<strong>{clean(hit.chunk.filename)}</strong>
          ／ {hit.chunk.page}ページ</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


faq = load_faq()
manual_signature = manual_file_signature()
manual_chunks, manual_warnings = cached_manuals(manual_signature)

st.markdown(
    """
    <div class="brandbar">
      <span class="brandmark">IT</span>
      <span>社内ヘルプデスク</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="page-intro">
      <h1>社内ITサポート</h1>
      <p>パソコンやネットワーク、アカウントなどのお困りごとをFAQとマニュアルから確認できます。</p>
      <div class="intro-meta">
        <span class="meta-pill">登録FAQ {len(faq)}件</span>
        <span class="meta-separator">|</span>
        <span class="meta-pill">登録PDF {len(manual_signature)}件</span>
        <span class="meta-separator">|</span>
        <span>デモ環境</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

search_tab, category_tab, list_tab = st.tabs(
    ["キーワード検索", "カテゴリから探す", "FAQ一覧"]
)

with search_tab:
    st.markdown('<div class="section-title">キーワードから調べる</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-description">現在の状況やエラーメッセージなどを入力してください。FAQと事前登録済みのPDFマニュアルを検索します。</p>',
        unsafe_allow_html=True,
    )

    with st.form("keyword_search_form", clear_on_submit=False):
        query = st.text_input(
            "問い合わせ内容",
            placeholder="例：VPNに接続できない",
        )
        submitted = st.form_submit_button("FAQ・マニュアルを検索", type="primary")

    st.markdown(
        '<div class="helper-note">検索例：パスワードを忘れた ／ メールが送信できない ／ 印刷ができない</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        if not query.strip():
            st.session_state.keyword_result = {"type": "blank", "query": ""}
        else:
            found = search_questions(query, faq)
            st.session_state.keyword_result = (
                {"type": "found", "indices": found, "query": query.strip(), "selected_index": None}
                if found
                else {"type": "not_found", "query": query.strip()}
            )

    keyword_result = st.session_state.get("keyword_result")
    if keyword_result:
        manual_hits = []
        if keyword_result["type"] == "found":
            st.markdown(
                '<div class="section-title" style="margin-top:25px">関連するFAQ</div>',
                unsafe_allow_html=True,
            )
            st.caption("近い内容の質問を最大3件表示しています。該当する質問を選んでください。")
            for candidate_index in keyword_result["indices"]:
                candidate = faq.iloc[candidate_index]
                with st.container(border=True):
                    st.markdown(f"**{clean(candidate['question'])}**")
                    st.caption(f"カテゴリ：{candidate['category']}")
                    if st.button("この質問の回答を見る", key=f"keyword_faq_{candidate_index}"):
                        st.session_state.keyword_result["selected_index"] = candidate_index
            selected_index = keyword_result.get("selected_index")
            if selected_index in keyword_result["indices"]:
                display_answer(faq.iloc[selected_index], label="選択したFAQの回答")
        elif keyword_result["type"] == "blank":
            st.warning("問い合わせ内容を入力してください。")
        if keyword_result["type"] != "blank":
            manual_hits = search_manuals(keyword_result.get("query", ""), manual_chunks)
            if manual_hits:
                st.markdown(
                    '<div class="section-title" style="margin-top:25px">関連するマニュアル</div>',
                    unsafe_allow_html=True,
                )
                st.caption("登録済みPDFから検索した文章の抜粋です。内容は参照元のページで確認してください。")
                for hit in manual_hits:
                    display_manual_hit(hit)

        if keyword_result["type"] == "not_found" and not manual_hits:
            st.markdown(
                """
                <div class="empty-result">
                  <div class="empty-result-title">該当するFAQ・マニュアルが見つかりませんでした</div>
                  <div class="empty-result-body">
                    別の言葉で検索するか、「カテゴリから探す」をお試しください。<br>
                    解決しない場合はIT担当者にお問い合わせください。
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

if manual_warnings:
    with st.expander("PDF読み込みのお知らせ"):
        for warning in manual_warnings:
            st.warning(warning)

with category_tab:
    st.markdown('<div class="section-title">カテゴリから調べる</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-description">お問い合わせの種類を選ぶと、関連する質問を一覧から探せます。</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        selected_category = st.selectbox(
            "カテゴリ", faq["category"].drop_duplicates().tolist()
        )
        filtered_faq = faq.loc[faq["category"] == selected_category]
        selected_question = st.selectbox(
            "お問い合わせ内容", filtered_faq["question"].tolist()
        )
        if st.button("対応方法を確認", key="show_selected_faq", type="primary"):
            selected_row = filtered_faq.loc[
                filtered_faq["question"] == selected_question
            ].iloc[0]
            st.session_state.category_result = {
                "question": selected_question,
                "id": str(selected_row["id"]),
            }

    category_result = st.session_state.get("category_result")
    if category_result and category_result["question"] == selected_question:
        matches = filtered_faq.loc[filtered_faq["id"] == category_result["id"]]
        if not matches.empty:
            display_answer(matches.iloc[0], label="FAQの回答")

with list_tab:
    st.markdown('<div class="section-title">登録済みFAQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-description">現在登録されているFAQを、カテゴリ別に絞り込んで確認できます。</p>',
        unsafe_allow_html=True,
    )
    view_category = st.selectbox(
        "カテゴリで絞り込む", ["すべて"] + faq["category"].drop_duplicates().tolist(),
        key="faq_list_category",
    )
    view = faq if view_category == "すべて" else faq.loc[faq["category"] == view_category]
    st.dataframe(
        view.rename(
            columns={
                "id": "No.",
                "category": "カテゴリ",
                "question": "質問",
                "answer": "対応方法",
                "source": "参照元",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=440,
    )

st.markdown(
    """
    <div class="app-footer">
      このアプリは社内ITサポートを想定したポートフォリオ用の試作品です。<br>
      FAQおよびPDFマニュアルはすべて架空の内容です。実在の企業の操作手順ではありません。
    </div>
    """,
    unsafe_allow_html=True,
)
