# AI-TextAnalyze

オフライン・CPU のみで動作する多言語テキスト解析デスクトップツール。
本文から固有表現（人名・国名・地名・組織）を抽出し、5言語間の翻訳を行います。

## 要件

- Python 3.10+
- Windows 10/11（pywebview / WebView2）
- CPU のみ（GPU 不要）
- ピーク RAM 2GB 以内

## セットアップ

```powershell
cd F:\Cursor\AI-TextAnalyze
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## モデル配置

`model/` 配下に以下を配置してください（詳細は [`model/README.md`](model/README.md)）。

| 種別 | フォルダ例 | 必須ファイル |
|------|-----------|-------------|
| NER | `model/ner-gliner-multi-medium-int8/` | `model.onnx`, `tokenizer.json`, `gliner_config.json` |
| MT | `model/mt-nllb-600m-ct2-int8/` | `model.bin`, `config.json`, `shared_vocabulary.txt` |

モデル未配置時も UI は起動しますが、該当機能ボタンは無効化されます。

## 起動

```powershell
python app.py
```

## テスト

```powershell
python -m pytest
python scripts/run-user-story-tests.py
.\scripts\build-portable.ps1
```

## ポータブル配布

`.\scripts\build-portable.ps1` の成果物（`dist/AI-TextAnalyze/`）トップは次の3つだけです。

| 名前 | 役割 |
|------|------|
| `AI-TextAnalyze.exe` | 起動 |
| `model/` | NER / MT モデル（必須・exe 横） |
| `runtime/` | 依存ランタイム（触らなくてよい） |

onefile は毎回展開で起動が遅いため、onedir + `runtime/` 分離を採用しています。

## ライセンス注意

- **NLLB-200** は CC-BY-NC（非商用）です。商用利用は不可。
- 本ツールは社内・個人利用前提で設計しています。

## ドキュメント

- [docs/SPEC.md](docs/SPEC.md) — 開発設計図（正本）
- [docs/00-index.md](docs/00-index.md) — ドキュメント索引
- [docs/user-stories/](docs/user-stories/) — All_Status ユーザーストーリー
