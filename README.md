# HLSearch

素数を用いた「シフト行列」に対する反例(条件を満たす組み合わせ)を探索するツールです。DFS(深さ優先探索)による全探索を、進捗表示・チェックポイント再開・GUI操作に対応させた形で実装しています。

## 概要

- `PRIMES` にリストアップされた1579以下の素数(249個)を階層(レベル)として扱います。
- 各階層 `level` では、素数 `p = primes[level]` に対して `0 <= i < p` のシフト値 `i` を1つ選びます。
- レベルごとに選んだシフト値の複数の否定条件(`~shift`)を `AND` で積み重ねていき、残った「候補列(ビットが立っている列)」の数を数えます。
- 指定した深さ(`depth`)まで到達した時点での候補列数を集計し、その中で最大となる値(`max_count`)を達成したシフト列の組み合わせをすべて探索・記録します。
- 各探索結果(`depth`, `count`, `key`)はsqliteデータベース(既定では`search_results.db`)にも逐次書き込まれ、後から`SELECT`で自由に集計・分析できます。
- 内部的には `numpy` の `bool` 配列を `uint64` にビットパックし、Numba でコンパイルした AND とポップカウント処理で、大きな列数(`cols = 3159`)でも高速に処理できるようにしています。

## 主な機能

- **反復DFS**: 再帰ではなく明示的なスタックを使った反復探索により、深い探索(`max_depth = 249`)でも再帰上限やオーバーヘッドの問題を回避しています。
- **枝刈り**: 現在の候補列数が既知の最大値を下回った時点で、それ以上探索しても意味がない枝を打ち切ります。
- **チェックポイント保存/再開**: 探索状態(スタック、キー、マスク、これまでの結果など)をJSON形式で定期的に保存し、途中から再開できます(`version: 2` フォーマット)。
- **探索結果のsqlite保存**: 探索中に深さ(`depth`)まで到達した各ノードの `depth` / `count` / `key` をsqliteデータベースへバッチ書き込みします。大量の結果もSQLで柔軟に検索・集計できます。
- **db内容を使った探索の高速化(prefix再利用)**: `state.run(depth=...)` を呼ぶと、チェックポイントから再開する場合を除き、db内に `depth-1` の探索結果があるかどうかを自動的に検索します。レコードが見つかった場合は、そのシフト列(`key`)を確定済みのprefixとして使い、まだ決まっていない最後の1階層だけを探索します(depth-1までの全階層を再度総当りしません)。同じ `depth-1` に複数件のレコード(同じ `max_count` を達成する複数の `key`)がある場合は、そのすべてをprefix候補として順番に探索します。該当レコードが無い場合は、従来通りdbを使わず先頭階層から全探索します。
- **進捗表示**: `tqdm` によるコンソール進捗バー表示と、ログファイルへの記録(`HLSearch.log`、ローテーション付き)に対応しています。
- **GUI(PySide6)**: 「探索する深さ」を指定して探索を開始し、進捗・最良結果をウィンドウ上で確認できるデスクトップアプリを同梱しています。

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
- 最良値を達成した件数(`results`)
- 最良値を達成したシフト列の一覧

### コードから直接実行する(例)

```python
from HLSearch import SearchConfig, build_shift_table, State

config = SearchConfig()
config.depth = 8  # 使用する素数の個数

shift_table = build_shift_table(config.primes[:config.depth], config.cols)
state = State(config, shift_table, max_depth=config.max_depth)
state.run(depth=config.depth)

print("最良値:", state.max_count)
print("最良値を達成した件数:", state.results)
print("最良値を達成したシフト列:", state.max_shifts)
```

探索結果は自動的に `config.results_db_file`(既定は `search_results.db`)へも書き込まれます。書き込みを無効にしたい場合は `State(..., db_path=False)` を指定してください。書き込み先を変えたい場合は `State(..., db_path="任意のパス.db")` を指定します。

```python
import sqlite3

conn = sqlite3.connect(config.results_db_file)
rows = conn.execute(
    "SELECT depth, count, key FROM search_results ORDER BY count DESC LIMIT 10"
).fetchall()
for depth, count, key in rows:
    print(depth, count, key)
```

### depthを1つずつ増やしながら探索する(dbの自動活用)

`state.run(depth=...)` は、`resume_from` を指定しない限り、実行のたびにdb内の `depth-1` のレコードを自動検索し、見つかればそれをprefixとして使って最後の1階層だけを探索します。そのため、`depth` を1つずつ増やしながら繰り返し実行すると、2回目以降は毎回全階層を総当りする必要がなく、探索が高速化されます。

```python
from HLSearch import SearchConfig, build_shift_table, State

config = SearchConfig()

for depth in range(1, 9):
    config.depth = depth
    shift_table = build_shift_table(config.primes[:config.depth], config.cols)
    state = State(config, shift_table, max_depth=config.max_depth)
    state.run(depth=config.depth)  # depth-1のdbレコードがあれば自動的にキーとして使う
    print(depth, state.max_count, state.results)
```

- `depth-1` のレコードがdbに無い場合(初回や、そのdepthをまだ探索していない場合)は、従来通りdbを使わず先頭階層から全探索します。
- `depth-1` に複数の最良キーが記録されている場合は、そのすべてを候補prefixとして順に探索します。あるprefixで見つかった `max_count` は、`State` インスタンス内で共有されるため、後続prefixの枝刈りにもそのまま活かされます。
- この自動検索は `resume_from` を指定した場合(チェックポイントからの再開)には行われません。チェックポイントの再開が優先されます。

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
| `cols` | 探索する列数 | 3159 |
| `progress_mininterval` | `tqdm` の最短更新間隔(秒) | 1.0 |
| `postfix_update_interval` | 進捗表示・コールバック呼び出しの間隔(ノード数) | 10000 |
| `shift_path_file` | 結果出力先ファイルパス | スクリプトと同じディレクトリの `shift_path.txt` |
| `results_db_file` | 探索結果(`depth`/`count`/`key`)を書き込むsqliteファイルパス | スクリプトと同じディレクトリの `search_results.db` |

## ファイル構成

- `HLSearch.py` — 探索ロジックとPySide6製GUIを含むメインスクリプト。
- `HLSearch.log` — 実行時に自動生成されるログファイル(ローテーション、最大10MB×3世代)。
- `shift_path.txt` — 結果出力先(設定で変更可能)。
- `search_results.db` — 探索結果(`depth`, `count`, `key`)を記録するsqliteファイル(`results_db_file`で変更可能、`db_path=False`で無効化可能)。テーブル名は `search_results`。
- チェックポイントファイル(任意のパス、JSON形式)。

## 注意事項

- `depth` は `len(primes)`(249)を超えられません。
- 探索は組み合わせ数が非常に大きくなるため、深さを増やすほど計算時間が急増します。長時間実行する場合はチェックポイント機能の利用を推奨します。
- GUIを起動すると、スクリプトと同じディレクトリに`HLSearch.log`が作成されます。探索の開始・完了・エラーはこのファイルとコンソールへ、詳細情報はログファイルへ記録されます。
- `state.run(depth=...)` はdb内の `depth-1` のレコードを自動的にprefixとして使うため、意図せず古い(別条件の)`search_results.db` を使い回すと、誤ったprefixから探索を始めてしまう可能性があります。`primes` や `cols` などの条件を変えて探索し直す場合は、`results_db_file` を別ファイルにするか、既存のdbを削除・退避してください。