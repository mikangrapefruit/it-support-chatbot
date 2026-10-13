
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity　

# ページの設定
st.set_page_config(
    page_title="社内ITサポート",
)

st.title("社内ITサポートチャットボット")

st.write("社内ITに関するお困りごとをサポートします。")
st.caption("※このアプリは架空の社内IT環境を想定した試作品です。")

# FAQデータの読み込み
faq = pd.read_csv("faq.csv", dtype={"id": str}).fillna("")

st.caption(f"現在 {len(faq)} 件のFAQを登録しています。")

# 検索方法をタブで切り替える
tab1, tab2 = st.tabs([
    "カテゴリから探す",
    "自由入力で探す"
])

# ① 選択式FAQ
with tab1:

    st.subheader("カテゴリから探す")

    categories = faq["category"].unique().tolist()

    selected_category = st.selectbox(
        "カテゴリを選択してください",
        categories
    )

    filtered_faq = faq[
        faq["category"] == selected_category
    ]

    selected_question = st.selectbox(
        "質問を選択してください",
        filtered_faq["question"].tolist()
    )

    if st.button("回答を見る"):

        result = filtered_faq[
            filtered_faq["question"] == selected_question
        ].iloc[0]

        with st.chat_message("user"):
            st.write(selected_question)

        with st.chat_message("assistant"):
            st.write("**解決方法**")
            st.write(result["answer"])
            st.caption(
                "参照元：" + result["source"] + "（架空）"
            )


# ② 自然文検索
with tab2:

    st.subheader("自由入力で質問する")

    question = st.text_input(
        "お困りごとを入力してください",
        placeholder="例：パスワードを忘れた"
    )

    if st.button("質問する"):

        if not question.strip():
            st.warning("質問を入力してください。")

        else:
            with st.chat_message("user"):
                st.write(question)

            # FAQを数値データに変換
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 3)
            )

            faq_vectors = vectorizer.fit_transform(
                faq["question"]
            )

            # 入力した質問を数値化
            question_vector = vectorizer.transform(
                [question.strip()]
            )

            # 類似度を計算
            similarities = cosine_similarity(
                question_vector,
                faq_vectors
            )[0]

            best_index = similarities.argmax()
            best_score = similarities[best_index]

            with st.chat_message("assistant"):

                if best_score >= 0.30:

                    result = faq.iloc[best_index]

                    st.success("関連するFAQが見つかりました。")

                    st.write("**関連する質問**")
                    st.write(result["question"])

                    st.write("**解決方法**")
                    st.write(result["answer"])

                    st.caption(
                        "参照元：" + result["source"] + "（架空）"
                    )

                else:
                    st.warning(
                        "該当するFAQが見つかりませんでした。"
                    )
                    st.write(
                        "お手数ですが、IT担当者にお問い合わせください。"
                    )


# ③ FAQ一覧（折りたたみ表示）
st.divider()

with st.expander("登録済みFAQ一覧を確認する"):
    st.dataframe(faq, hide_index=True)

st.caption("社内ITサポートチャットボット | 試作版")
