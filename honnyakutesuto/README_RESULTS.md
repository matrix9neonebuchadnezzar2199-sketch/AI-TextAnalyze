# 翻訳テスト結果サマリ（honnyakutesuto）

実施日: 2026-07-13  
モデル: NLLB distilled 1.3B CT2 int8  
対象: RU / ZH / KO / EN（PDF→抽出テキスト）

## ループ

| Round | 主な修正 | 結果 |
|------|---------|------|
| 1 | ベースライン | KO/ZH に深刻な反復ループ。EN に英語断片。ZH/KO NER=0 |
| 2 | `no_repeat_ngram_size=3`, PDF空白正規化, ページ番号リテラル | KO 反復ほぼ解消。ZH 反復改善 |
| 3 | `www3.` URL保護, puma置換, 「ほら」除去, CJK NERラベル | KO URL保全。EN puma誤訳（熊）解消。ZH NERは依然0 |
| 4 | ラテン改行をスペース結合 | EN の単語連結は改善。固有名詞の英語残留・文順序乱れは残存 |

## 停止判断

これ以上の前処理・デコード調整では、次の残差は **モデル能力限界** と判断し停止。

残る限界（コードでは実質解消不可）:
- 中国語人名（秦雷→「雷」）や組織名の誤訳・欠落
- GLiNER ONNX int8 の中韓エンティティ抽出がほぼ効かない
- 固有地名の英語コピー残留（Sonoran Desert 等）
- PDF由来のキャプション・ページ番号・レイアウトノイズ
- 600M/1.3B 共通の語彙・長文ばらつき

## 成果物

- 抽出: `honnyakutesuto/_extracted/`
- 翻訳・NER: `honnyakutesuto/_results/`
- 再現: `python scripts/run_honnyaku_tests.py --mt nllb-1.3b --ner`

## コード変更（本セッション）

- `backend/mt_engine.py` — PDF前処理、反復抑制、URL、改行結合
- `backend/ner_engine.py` — CJK前処理・閾値・補助ラベル
- `scripts/run_honnyaku_tests.py` — バッチ評価
- `tests/test_mt_engine.py` — 回帰テスト追加
