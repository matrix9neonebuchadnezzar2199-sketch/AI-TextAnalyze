# AI-TextAnalyze ユーザーストーリー管理

全機能のユーザーストーリーとテスト結果を CSV で管理する（All_Status 手法）。

## ファイル

| ファイル | 用途 |
|----------|------|
| [user-stories-matrix.csv](./user-stories-matrix.csv) | ストーリーマトリクス（生成物） |
| [test-results.csv](./test-results.csv) | テスト実行ログ（自動生成） |
| [summary.md](./summary.md) | 集計サマリ（自動生成） |

## 正本（コード）

ストーリー定義: [`scripts/user_story_catalog.py`](../scripts/user_story_catalog.py)

## 実行

```powershell
cd F:\Cursor\AI-TextAnalyze
python -m pytest
python scripts/run-user-story-tests.py
```

## メモリ検証

各機能実行時のピーク RAM は 2GB 以内であること。手順は [`docs/memory-verification.md`](../memory-verification.md)。
