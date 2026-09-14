# HLSearch

素数を用いた「シフト行列」に対する反例(条件を満たす組み合わせ)を探索するツールです。DFS(深さ優先探索)による全探索を、進捗表示・チェックポイント再開・GUI操作に対応させた形で実装しています。

## 概要

- `PRIMES` にリストアップされた1579以下の素数(249個)を階層(レベル)として扱います。
- 各階層 `level` では、素数 `p = primes[level]` に対して `0 <= i < p` のシフト値 `i` を1つ選びます。
- レベルごとに選んだシフト値の複数の否定条件(`~shift`)を `AND` で積み重ねていき、残った「候補列(ビットが立っている列)」の数を数えます。
- 指定した深さ(`depth`)まで到達した時点での候補列数が「最大値」または「目標値(`target`)」と一致するシフト列の組み合わせを探索・記録します。
- 内部的には `numpy` の `bool` 配列を `uint64` にビットパックし、Numba でコンパイルした AND とポップカウント処理で、大きな列数(`cols = 3159`)でも高速に処理できるようにしています。

## 主な機能

- **反復DFS**: 再帰ではなく明示的なスタックを使った反復探索により、深い探索(`max_depth = 249`)でも再帰上限やオーバーヘッドの問題を回避しています。
- **枝刈り**: 現在の候補列数が既知の最大値を下回った時点で、それ以上探索しても意味がない枝を打ち切ります。
- **チェックポイント保存/再開**: 探索状態(スタック、キー、マスク、これまでの結果など)をJSON形式で定期的に保存し、途中から再開できます(`version: 2` フォーマット)。
- **進捗表示**: `tqdm` によるコンソール進捗バー表示と、ログファイルへの記録(`HLSearch.log`、ローテーション付き)に対応しています。
- **GUI(PySide6)**: 「探索する深さ」を指定して探索を開始し、進捗・最良結果・目標到達結果をウィンドウ上で確認できるデスクトップアプリを同梱しています。

## 必要要件

- Python 3.10 以上(型ヒントに `X | None` などの構文を使用)
- 依存パッケージ:
  - `numpy`
  - `numba`
  - `tqdm`
  - `PySide6`

```bash
pip install numpy numba tqdm PySide6
```

初回の探索時には、ビットマスクの積集合とポップカウントを担う Numba
カーネルのコンパイルが一度だけ行われます。以降の DFS ノードでは、
Python ループではなく nopython モードで実行されます。

## 使い方

### GUIから実行する

```bash
python HLSearch.py
```

ウィンドウが起動したら、「探索する深さ(使用する素数の個数)」を指定して「探索開始」ボタンを押します。探索はバックグラウンドスレッド(`SearchWorker`)で実行されるため、GUIがフリーズすることはありません。完了すると、以下が結果欄に表示されます。

- 総ノード数
- 最良値(`max_count`)
- 目標値(`target`)への到達回数
- 最良値・目標値をそれぞれ達成したシフト列の一覧

### コードから直接実行する(例)

```python
from HLSearch import SearchConfig, build_shift_table, State

config = SearchConfig()
config.depth = 8  # 使用する素数の個数

shift_table = build_shift_table(config.primes[:config.depth], config.cols)
state = State(config, shift_table, max_depth=config.max_depth, target=config.target)
state.run(depth=config.depth)

print("最良値:", state.max_count)
print("目標到達回数:", state.target_results)
```

### チェックポイントからの再開

```python
state.run(depth=config.depth, resume_from="checkpoint.json")
```

`State` を `checkpoint_path=` 付きで生成しておくと、`checkpoint_interval`(既定1000ノード)ごとに自動でチェックポイントが保存されます。

## 主要パラメータ(`SearchConfig`)

| 属性 | 説明 | 既定値 |
| --- | --- | --- |
| `primes` | 探索対象の素数リスト | 1579以下の素数(249個) |
| `depth` | 実際に使用する素数の個数(探索の深さ) | 8 |
| `max_depth` | 深さの理論上限 | 249 |
| `target` | `depth == max_depth` 時の打ち切り目標値 | 447 |
| `cols` | 探索する列数 | 3159 |
| `progress_mininterval` | `tqdm` の最短更新間隔(秒) | 1.0 |
| `postfix_update_interval` | 進捗表示・コールバック呼び出しの間隔(ノード数) | 10000 |
| `shift_path_file` | 結果出力先ファイルパス | スクリプトと同じディレクトリの `shift_path.txt` |

## ファイル構成

- `HLSearch.py` — 探索ロジックとPySide6製GUIを含むメインスクリプト。
- `HLSearch.log` — 実行時に自動生成されるログファイル(ローテーション、最大10MB×3世代)。
- `shift_path.txt` — 結果出力先(設定で変更可能)。
- チェックポイントファイル(任意のパス、JSON形式)。

## 注意事項

- `depth` は `len(primes)`(249)を超えられません。
- 探索は組み合わせ数が非常に大きくなるため、深さを増やすほど計算時間が急増します。長時間実行する場合はチェックポイント機能の利用を推奨します。