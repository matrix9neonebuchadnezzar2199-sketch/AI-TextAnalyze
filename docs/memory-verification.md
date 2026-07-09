# メモリ検証手順

AI-TextAnalyze は **ピーク RAM 2GB 以内** が絶対制約です。

## 前提

- モデルを `model/` に配置済みであること
- Windows タスクマネージャーで `python.exe` のメモリを監視

## 手順

1. `python app.py` で起動（モデル未ロード時の RAM を記録）
2. 短文（〜500文字）で「キーワード抽出」を実行 → ピーク RAM を記録 → NER 完了後に RAM が下がることを確認
3. 同じ本文で「翻訳実行」→ ピーク RAM を記録 → MT 完了後に RAM が下がることを確認
4. 抽出直後に翻訳を実行し、**両モデルが同時常駐していない**ことを確認（排他ロード）

## 合格基準

| フェーズ | 目安 |
|---------|------|
| アイドル（モデル未ロード） | < 200 MB |
| NER 実行ピーク | < 1.0 GB |
| MT 実行ピーク | < 1.5 GB |
| いずれの時点でも | **< 2.0 GB** |

## 自動計測（任意）

```powershell
python -c "import psutil, os; print(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 'MB')"
```

※ `psutil` は開発用。本番 `requirements.txt` には含めない。
