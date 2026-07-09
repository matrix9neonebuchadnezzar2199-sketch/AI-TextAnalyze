# モデル配置ガイド

AI-TextAnalyze は `model/` 配下のサブフォルダを起動時に自動検知します。
**実行時のダウンロードは行いません。** 事前にモデルを配置してください。

## NER（GLiNER ONNX int8）

配置先例: `model/ner-gliner-multi-medium-int8/`

**入手元**: [onnx-community/gliner_multi-v2.1](https://huggingface.co/onnx-community/gliner_multi-v2.1)（`onnx/model_int8.onnx`）

一括ダウンロード:

```powershell
python scripts/download-models.py
```

必須ファイル:
- `model.onnx` — ONNX int8 量子化モデル
- `tokenizer.json` — トークナイザ
- `gliner_config.json` — GLiNER 設定

抽出ラベル（固定）: `Person`, `Country`, `City`, `Organization`

## MT（NLLB CTranslate2 int8）

UI 上部で **600M / 1.3B** を切り替え可能。選択時にモデルロード画面が表示されます。

| バリアント | 配置先 | 入手元 |
|-----------|--------|--------|
| distilled 600M | `model/mt-nllb-600m-ct2-int8/` | [Tushe/nllb-200-600M-ct2-int8](https://huggingface.co/Tushe/nllb-200-600M-ct2-int8) |
| distilled 1.3B | `model/mt-nllb-1.3b-ct2-int8/` | [OpenNMT/nllb-200-distilled-1.3B-ct2-int8](https://huggingface.co/OpenNMT/nllb-200-distilled-1.3B-ct2-int8) |

```powershell
# 600M のみ（従来どおり）
python scripts/download-models.py --mt nllb-600m

# 1.3B を追加
python scripts/download-models.py --mt nllb-1.3b
```

必須ファイル（各フォルダ）:
- `model.bin`, `config.json`, `shared_vocabulary.*`
- `hf-tokenizer/` — サイズ対応の NLLB HF トークナイザ（スクリプトが自動取得）

## ライセンス

- NLLB-200: **CC-BY-NC**（非商用）。商用転用不可。
- GLiNER: モデル配布元のライセンスに従うこと。

## メモリ目安

| モデル | RAM 目安 |
|--------|---------|
| NER | 0.5〜1.0 GB |
| MT 600M | 1.0〜1.5 GB |
| MT 1.3B | 約 2〜2.5 GB（メモリ余裕が必要） |

NER と MT は**同時常駐しません**（排他ロード）。
