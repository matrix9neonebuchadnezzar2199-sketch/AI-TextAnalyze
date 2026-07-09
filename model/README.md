# モデル配置ガイド

AI-TextAnalyze は `model/` 配下のサブフォルダを起動時に自動検知します。
**実行時のダウンロードは行いません。** 事前にモデルを配置してください。

## NER（GLiNER ONNX int8）

配置先例: `model/ner-gliner-multi-medium-int8/`

必須ファイル:
- `model.onnx` — ONNX int8 量子化モデル
- `tokenizer.json` — トークナイザ
- `gliner_config.json` — GLiNER 設定

抽出ラベル（固定）: `Person`, `Country`, `City`, `Organization`

## MT（NLLB CTranslate2 int8）

配置先例: `model/mt-nllb-600m-ct2-int8/`

必須ファイル:
- `model.bin`
- `config.json`
- `shared_vocabulary.txt`（または `shared_vocabulary` プレフィックス）

## ライセンス

- NLLB-200: **CC-BY-NC**（非商用）。商用転用不可。
- GLiNER: モデル配布元のライセンスに従うこと。

## メモリ目安

| モデル | RAM 目安 |
|--------|---------|
| NER | 0.5〜1.0 GB |
| MT | 1.0〜1.5 GB |

NER と MT は**同時常駐しません**（排他ロード）。
