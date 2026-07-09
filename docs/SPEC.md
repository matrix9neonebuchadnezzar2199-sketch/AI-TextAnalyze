# AI-TextAnalyze — 開発設計図 / 仕様書

## 0. これは何か
オフライン（完全ローカル）で動作する多言語テキスト解析デスクトップツール。
貼り付けた本文から人名・地名・国名・組織などの固有表現を自動抽出し、
同時に多言語翻訳を行う。GPU不要・CPUのみ・低メモリ環境で動くことを最優先とする。

## 1. 絶対制約（MUST。これを破る実装は不可）
- **GPUを一切使用しない。** CPU推論のみ。CUDA依存パッケージを入れない。
- **ピークメモリ 2GB 以内。** これを超える実装・モデルロードは禁止。
- **完全オフライン。** 実行時に外部APIやモデルの自動ダウンロードを行わない。
- **モデルは `/model` フォルダに配置し、起動時に自動検知する。**
- **NERモデルと翻訳モデルを同時にメモリ常駐させない（排他ロード必須）。**
  抽出時はNERのみ、翻訳時は翻訳モデルのみをロードし、使用後に解放する。
- 対応言語：日本語(ja)・英語(en)・中国語(zh)・ロシア語(ru)・韓国語(ko)。

## 2. 採用モデル（確定。勝手に変更しない）
### NER（キーワード抽出）
- **GLiNER multilingual medium（mDeBERTa-base系）を ONNX int8 量子化して使用。**
- 抽出ラベルは指定式：`Person`（人名）, `Country`（国名）, `City`（都市/地名）, `Organization`（組織）。
- 国名と地名を区別して出力できる点を活かす。
- 実行時RAM目安 0.5〜1.0GB。

### 翻訳
- **NLLB-200 distilled 600M を CTranslate2 int8 で使用。**
- 5言語を1モデルで相互翻訳。翻訳元は自動判定可、翻訳後の初期値は日本語(ja)。
- 実行時RAM目安 1.0〜1.5GB。
- CPUで1文あたり数秒かかりうるため、文単位に分割して逐次処理し進捗表示する。
- ※NLLBはCC-BY-NC（非商用）。本ツールは社内・個人利用前提のため採用可。商用転用は不可。

## 3. アーキテクチャ

```
[ フロントエンド：HTML/CSS/JS (WebView) ]
│  ローカル通信 (pywebview bridge)
[ バックエンド：Python ]
├─ model_manager  … /model 自動検知・排他ロード・解放
├─ ner_engine     … GLiNER(ONNX) 推論
├─ mt_engine      … NLLB(CTranslate2) 推論
├─ pdf_reader     … PDF/txt からテキスト抽出
└─ lang_detect    … 翻訳元言語の自動判定
```

- UIとPythonの接続は **pywebview** を第一候補とする（単体デスクトップアプリ化しやすく、
  ローカルポートを開かないので社内環境で安全）。

## 4. ディレクトリ構成

```
AI-TextAnalyze/
├─ docs/
│  ├─ SPEC.md                 ← 本設計図（正本）
│  ├─ 00-index.md
│  └─ user-stories/
├─ .cursor/rules/
│  └─ ai-textanalyze.mdc      ← Cursor 常時参照ルール
├─ app.py                     ← エントリポイント（WebView起動）
├─ requirements.txt
├─ backend/
│  ├─ __init__.py
│  ├─ model_manager.py        ← /model 検知・排他ロード
│  ├─ ner_engine.py           ← GLiNER ONNX 推論
│  ├─ mt_engine.py            ← NLLB CTranslate2 推論
│  ├─ pdf_reader.py           ← PDF/txt 読み込み
│  ├─ lang_detect.py          ← 言語判定
│  └─ api.py                  ← フロントに公開する関数群（bridge）
├─ frontend/
│  ├─ index.html
│  ├─ style.css
│  └─ app.js
└─ model/                     ← モデル配置先（同梱 or 別配布）
   ├─ ner-gliner-multi-medium-int8/
   │   ├─ model.onnx
   │   ├─ tokenizer.json
   │   └─ gliner_config.json
   └─ mt-nllb-600m-ct2-int8/
       ├─ model.bin
       ├─ config.json
       └─ shared_vocabulary.txt
```

## 5. モデル自動検知の仕様（model_manager.py）
起動時に `model/` 配下の各サブフォルダを走査し、ファイル構成で種別を判定する。
- CTranslate2翻訳モデル判定：`model.bin` + `config.json` + `shared_vocabulary*` が存在。
- GLiNER NERモデル判定：`*.onnx` + `tokenizer.json`（+ `gliner_config.json`）が存在。
判定結果（モデル名）をUIステータスバーに表示。
必要モデルが欠けている場合はUIに警告を出し、該当機能ボタンを無効化する。

### 排他ロードの実装ルール
- `ModelManager` はどちらか片方のエンジンのみ保持する。
- `load_ner()` を呼ぶ前に `unload_mt()` を実行、`load_mt()` の前に `unload_ner()` を実行。
- 解放時は明示的に参照を切り `gc.collect()` を呼ぶ。ONNX/CT2セッションも明示破棄。
- どの時点でもメモリに載るモデルは1つだけであることをコードで保証する。

## 6. UI仕様（frontend）
3ペイン構成：左=本文 / 中=翻訳 / 右=抽出キーワード。
- 上部ツールバー：タイトル「AI-TextAnalyze」、PDF/テキスト添付、キーワード抽出、Light/Darkトグル。
- 本文ペイン：貼り付け＋PDF読込。文字数と自動判定言語を表示。
- 翻訳ペイン：翻訳元言語(自動判定+5言語)・入替ボタン・翻訳後言語(初期=日本語)・翻訳実行ボタン。
  長文は進捗をステータスバーに表示。
- キーワードペイン：
  - デフォルトはタイプ別バッジ表示（人名/国名/地名/組織）＋フィルタチップ。
  - 「リスト表示」で **キーワード＋改行の繰り返しの単純リスト** に切替（仕様の核）。
  - 「リストをコピー」で単純リスト形式をクリップボードへ。
- テーマ：Light/Dark切替可。プロフェッショナルな配色（CSS変数を使用）。

## 7. バックエンドAPI（api.py がフロントに公開する関数）
| 関数 | 入力 | 出力 | 備考 |
|---|---|---|---|
| `get_model_status()` | なし | 検知したNER/MTモデル名, 利用可否 | 起動時に呼ぶ |
| `pick_and_read_file()` | なし | 抽出テキスト | ファイルダイアログ→PDF/txt対応 |
| `detect_language(text)` | text | 言語コード | 翻訳元自動判定用 |
| `extract_keywords(text)` | text | `[{term,type,freq}]` | NERを排他ロードして実行 |
| `translate(text, src, tgt)` | text,言語 | 翻訳文 | 翻訳を排他ロードして実行 |

- `type` は `per`/`country`/`city`/`org` を返す。UI側でバッジ色分け。
- 抽出結果は出現頻度で集約し重複除去する。

## 8. 依存パッケージ（requirements.txt に相当・CPU版のみ）
- `onnxruntime`（**GPU版 onnxruntime-gpu は禁止**）
- `ctranslate2`
- `sentencepiece`（NLLBトークナイズ用）
- `tokenizers`（GLiNERトークナイズ用）
- `pywebview`
- `pypdf`（PDF読み込み）
- `langid`（言語自動判定用・オフライン軽量）
- `numpy`

※ torch は推論に載せない方針（ONNX/CT2で完結させ、メモリを節約する）。

## 9. パフォーマンス方針
- 翻訳は文分割 → 逐次翻訳 → 結合。1文ずつ進捗更新。
- ONNX Runtimeはスレッド数をCPUコアに合わせて設定。intra_op_num_threadsを調整可能に。
- モデルロードは初回のみ重いので、ロード中はUIにスピナー/ステータス表示。
- 例外時（モデル未検知・メモリ不足・PDF解析失敗）は必ずUIに分かるメッセージを返す。

## 10. 配布形態
- ポータブル運用。`app.exe`（PyInstaller）＋隣に `model/` フォルダ＋`frontend/`。
- モデルはEXEに埋め込まず外部フォルダ配置（差し替え・更新しやすくするため）。
- 完全な単一EXEは目指さない（サイズと保守性の観点から）。

## 11. 実装の進め方
1. `model_manager.py` … /model検知＋排他ロードの骨格を先に作る（最重要）。
2. `ner_engine.py` … GLiNER ONNX推論。ダミー入力で動作確認。
3. `mt_engine.py` … NLLB CTranslate2推論。文分割・逐次翻訳を実装。
4. `pdf_reader.py` / `lang_detect.py`。
5. `api.py` … 上記を排他ロード制御込みで公開。
6. `frontend/` … モックを実APIに接続。
7. メモリ計測（各機能実行時のピークRAM）を行い、2GB以内を検証。
8. PyInstaller でポータブル化。

## 12. コーディング規約
- 上記「絶対制約」を破るコードは書かない。特にGPU依存・自動ダウンロード・両モデル同時常駐は禁止。
- メモリを意識し、大きなオブジェクトは使用後に明示解放する。
- 各エンジンは「ロード → 推論 → アンロード」を1つのクラスにまとめ、状態を明確にする。
- 例外は握りつぶさず、UIに返すメッセージへ変換する。
- コメントは日本語可。関数には型ヒントを付ける。
