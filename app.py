"""社内ITサポート FAQ検索アプリ
"""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="社内ITサポート | FAQ検索",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# このアプリ専用のスタイル。既存のFAQ検索ロジックには影響しません。
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
    """チャットの吹き出しではなく、社内FAQ風の結果パネルを表示する。"""
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


def search_question(question: str, data: pd.DataFrame) -> int | None:
    """元のアプリと同じ日本語文字n-gram検索と閾値0.30。"""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    vectors = vectorizer.fit_transform(data["question"].astype(str))
    question_vector = vectorizer.transform([question.strip()])
    scores = cosine_similarity(question_vector, vectors)[0]
    best_index = int(scores.argmax())
    return best_index if scores[best_index] >= 0.30 else None


faq = load_faq()

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
      <p>パソコンやネットワーク、アカウントなどのお困りごとをFAQから確認できます。</p>
      <div class="intro-meta">
        <span class="meta-pill">登録FAQ {len(faq)}件</span>
        <span class="meta-separator">|</span>
        <span>{faq['category'].nunique()}カテゴリ</span>
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
        '<p class="section-description">現在の状況やエラーメッセージなどを入力してください。登録済みのFAQから近い内容を検索します。</p>',
        unsafe_allow_html=True,
    )

    with st.form("keyword_search_form", clear_on_submit=False):
        query = st.text_input(
            "問い合わせ内容",
            placeholder="例：VPNに接続できない",
        )
        submitted = st.form_submit_button("FAQを検索", type="primary")

    st.markdown(
        '<div class="helper-note">検索例：パスワードを忘れた ／ メールが送信できない ／ 印刷ができない</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        if not query.strip():
            st.session_state.keyword_result = {"type": "blank"}
        else:
            found = search_question(query, faq)
            st.session_state.keyword_result = (
                {"type": "found", "index": found}
                if found is not None
                else {"type": "not_found"}
            )

    keyword_result = st.session_state.get("keyword_result")
    if keyword_result:
        if keyword_result["type"] == "found":
            display_answer(faq.iloc[keyword_result["index"]])
        elif keyword_result["type"] == "blank":
            st.warning("問い合わせ内容を入力してください。")
        else:
            st.markdown(
                """
                <div class="empty-result">
                  <div class="empty-result-title">該当するFAQが見つかりませんでした</div>
                  <div class="empty-result-body">
                    別の言葉で検索するか、「カテゴリから探す」をお試しください。<br>
                    解決しない場合はIT担当者にお問い合わせください。
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
      FAQおよび参照資料はすべて架空の内容です。
    </div>
    """,
    unsafe_allow_html=True,
)
