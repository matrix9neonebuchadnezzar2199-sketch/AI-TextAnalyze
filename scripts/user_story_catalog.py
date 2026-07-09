"""User story catalog — source of truth for All_Status matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserStory:
    story_id: str
    epic: str
    feature: str
    route: str
    persona: str
    user_story: str
    precondition: str
    steps: str
    expected_behavior: str
    apis: str
    impl_status: str
    test_method: str


STORIES: list[UserStory] = [
    UserStory(
        story_id="CORE-001",
        epic="起動",
        feature="アプリ起動",
        route="/",
        persona="アナリスト",
        user_story="オフライン環境でデスクトップアプリを起動したい",
        precondition="Python 依存がインストール済み",
        steps="1. python app.py を実行\n2. WebView が表示される",
        expected_behavior="3ペイン UI が表示されステータスバーにモデル検知結果が出る",
        apis="get_model_status",
        impl_status="implemented",
        test_method="manual",
    ),
    UserStory(
        story_id="SRC-001",
        epic="本文",
        feature="テキスト入力",
        route="/",
        persona="アナリスト",
        user_story="本文を貼り付けて文字数と言語を確認したい",
        precondition="アプリ起動済み",
        steps="1. 本文ペインにテキストを貼り付け",
        expected_behavior="文字数と自動判定言語が更新される",
        apis="detect_language",
        impl_status="implemented",
        test_method="unit",
    ),
    UserStory(
        story_id="SRC-002",
        epic="本文",
        feature="ファイル読込",
        route="/",
        persona="アナリスト",
        user_story="PDF/txt を添付して本文に読み込みたい",
        precondition="アプリ起動済み",
        steps="1. PDF/テキスト添付をクリック\n2. ファイルを選択",
        expected_behavior="本文ペインに抽出テキストが表示される",
        apis="pick_and_read_file,read_text_file",
        impl_status="implemented",
        test_method="unit",
    ),
    UserStory(
        story_id="NER-001",
        epic="キーワード",
        feature="固有表現抽出",
        route="/",
        persona="アナリスト",
        user_story="人名・国名・地名・組織を抽出したい",
        precondition="NER モデルが model/ に配置済み",
        steps="1. 本文を入力\n2. キーワード抽出をクリック",
        expected_behavior="per/country/city/org バッジ付きキーワード一覧が表示される",
        apis="extract_keywords",
        impl_status="implemented",
        test_method="manual",
    ),
    UserStory(
        story_id="NER-002",
        epic="キーワード",
        feature="リスト表示・コピー",
        route="/",
        persona="アナリスト",
        user_story="キーワードを改行区切りリストでコピーしたい",
        precondition="抽出結果がある",
        steps="1. リスト表示に切替\n2. リストをコピー",
        expected_behavior="キーワード+改行形式がクリップボードに入る",
        apis="",
        impl_status="implemented",
        test_method="manual",
    ),
    UserStory(
        story_id="MT-001",
        epic="翻訳",
        feature="多言語翻訳",
        route="/",
        persona="アナリスト",
        user_story="5言語間で本文を翻訳したい",
        precondition="MT モデルが model/ に配置済み",
        steps="1. 翻訳元・翻訳後を選択\n2. 翻訳実行",
        expected_behavior="翻訳ペインに訳文が表示され進捗がステータスバーに出る",
        apis="translate",
        impl_status="implemented",
        test_method="manual",
    ),
    UserStory(
        story_id="SYS-001",
        epic="システム",
        feature="排他ロード",
        route="/",
        persona="運用者",
        user_story="NER と MT が同時にメモリに載らないことを保証したい",
        precondition="両モデル配置済み",
        steps="1. 抽出実行\n2. 翻訳実行",
        expected_behavior="各実行時に片方のモデルのみロードされ完了後解放される",
        apis="model_manager",
        impl_status="implemented",
        test_method="unit",
    ),
    UserStory(
        story_id="SYS-002",
        epic="システム",
        feature="モデル未検知",
        route="/",
        persona="運用者",
        user_story="モデル未配置でもクラッシュせず警告したい",
        precondition="model/ が空",
        steps="1. アプリ起動",
        expected_behavior="該当ボタンが無効化されステータスに未検出と表示",
        apis="get_model_status",
        impl_status="implemented",
        test_method="unit",
    ),
]
