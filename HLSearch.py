#HLSearch.py
import csv
import sys
import os
import logging
import logging.handlers
import json
import numpy as np
from numba import njit
from dataclasses import dataclass, field
import time
from pathlib import Path
import itertools
from typing import Sequence
import multiprocessing
from tqdm import tqdm
import ctypes
from numpy.typing import NDArray


from PySide6.QtCore import Qt, QTimer, QModelIndex, QThread, Signal
from PySide6.QtGui import QAction, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QFileDialog,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableView,
    QPlainTextEdit,
    QStyledItemDelegate,
    QWidget,
)
from PySide6.QtCore import QAbstractTableModel
import numpy as np
import time

# 2,3,5,7,...,1579 の素数リスト(249個)
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
    107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
    229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479,
    487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
    631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769,
    773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
    1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193,
    1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307,
    1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459,
    1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579,
]
ROWS = len(PRIMES)  # 249 rows
COLS = 3159
TARGET = 447
WORD_BITS = 64  # uint64 1語あたりのビット数(shift_table のパック単位)
PARAMS = [
    [i for i in range(prime)] for prime in list(PRIMES)        
]


class SearchConfig:
    """探索処理に必要な設定をまとめた構成体。

    Attributes:
        primes: 探索対象の素数リスト。デフォルトでは 1579 以下の素数を生成する。
        depth: 深さとして使う素数の数。
        max_depth: 深さの上限。
        target: `depth == max_depth` のときの打ち切り目標値.
        cols: 列数。
        progress_mininterval: tqdm の最短更新間隔。
        postfix_update_interval: postfix 更新の頻度。
        shift_path_file: 出力ファイルパス.
    """
    primes: list[int] = PRIMES
    params: list[list[int]] = PARAMS
    depth: int = 8
    max_depth: int = 249
    target: int = 447
    cols: int = COLS
    progress_mininterval: float = 1.0
    postfix_update_interval: int = 10000
    shift_path_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_path.txt")

    def __post_init__(self) -> None:
        """設定値の整合性を早期に検証する(実行時ではなく構築時に失敗させる)。"""
        if self.cols <= 0:
            raise ValueError(f"cols は正の整数である必要があります: cols={self.cols}")
        if self.depth < 0:
            raise ValueError(f"depth は0以上である必要があります: depth={self.depth}")
        if self.depth > len(self.primes):
            raise ValueError(
                f"depth={self.depth} が primes の要素数({len(self.primes)})を超えています"
            )
        if self.postfix_update_interval <= 0:
            raise ValueError(
                f"postfix_update_interval は正の整数である必要があります: "
                f"postfix_update_interval={self.postfix_update_interval}"
            )

cfg = SearchConfig()
shift_path_file: str = cfg.shift_path_file

# --- logging設定 ---
logger = logging.getLogger(__name__)

def setup_logging(base_dir: str | os.PathLike[str], console_level: str="INFO") -> str:
    """コンソールとファイルの両方にログを出力するよう設定する。"""
    log_path = os.path.join(base_dir, "HLSearch.log")

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 二重登録防止(再実行・再インポート対策)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"ログファイルを作成しました: {log_path}")
    return log_path


def shift_array(arr: NDArray[np.bool_], k: int) -> NDArray[np.bool_]:
    """
    配列を右にk個シフトする(numpy版)。先頭k要素は0埋めし、末尾のk要素は捨てる。
    """
    n = len(arr)
    if k <= 0:
        return arr.copy()
    if k >= n:
        return np.zeros(n, dtype=arr.dtype)
    result = np.empty(n, dtype=arr.dtype)
    result[:k] = 0
    result[k:] = arr[: n - k]
    return result
 

@njit
def intersect_masks_and_count(
    base_mask: NDArray[np.uint64],
    row_complement: NDArray[np.uint64],
) -> tuple[NDArray[np.uint64], int]:
    """二つのパック済みマスクをANDし、残る候補数を数える。"""
    node_mask = np.empty(base_mask.size, dtype=np.uint64)
    count = 0
    for index in range(base_mask.size):
        value = base_mask[index] & row_complement[index]
        node_mask[index] = value
        while value:
            value &= value - np.uint64(1)
            count += 1
    return node_mask, count


def build_base_rows(primes: Sequence[int], cols: int = cfg.cols) -> NDArray[np.bool_]:
    """指定した素数リストから各階層の基底行を生成する。

    各要素は `bool((idx % p) == 1)` を保持し、探索では「0かどうか」だけを
    判定する。`bool` 型にすることで 1 要素あたりのメモリ使用量を抑え、
    シフトや AND 演算を高速化する。

    Args:
        primes: 基底行を作る素数一覧。
        cols: 配列の列数。

    Returns:
        shape=(len(primes), cols) の bool 配列。
    """
    idx = np.arange(1, cols + 1)
    return np.array([(idx % p == 1) for p in primes])


def _pack_masks(masks: NDArray[np.bool_]) -> NDArray[np.uint64]:
    """bool マスクを列方向に uint64 ワードへパックする。"""
    packed_bytes = np.packbits(masks, axis=-1, bitorder="little")
    padding = (-packed_bytes.shape[-1]) % (WORD_BITS // 8)
    if padding:
        pad_width = [(0, 0)] * packed_bytes.ndim
        pad_width[-1] = (0, padding)
        packed_bytes = np.pad(packed_bytes, pad_width)
    return np.ascontiguousarray(packed_bytes).view(np.uint64)


def _packed_width(cols: int) -> int:
    return (cols + WORD_BITS - 1) // WORD_BITS


def build_shift_table(primes: Sequence[int], cols: int = cfg.cols) -> list[NDArray[np.uint64]]:
    """各階層ごとのシフト候補テーブルを事前生成する。

    これにより探索時に毎回 `shift_array()` と `~` 演算を行わず、
    事前に補集合を計算済みの配列をそのまま使える。

    Args:
        primes: 使用する素数のリスト。
        cols: 列数。

    Returns:
        `shift_table[level][shift]` が、level 段目におけるシフト値 `shift`
        に対応する補集合行を表す uint64 パック配列。
    """
    base_rows = build_base_rows(primes, cols)
    shift_table: list[NDArray[np.uint64]] = []
    for level, p in enumerate(primes):
        row = base_rows[level]
        shifted_complement = np.empty((p, cols), dtype=bool)
        for k in range(p):
            shifted_complement[k] = ~shift_array(row, k)
        shift_table.append(_pack_masks(shifted_complement))
    return shift_table


def build_initial_zero_mask(cols: int = cfg.cols) -> NDArray[np.uint64]:
    """探索開始時点(まだ何も確定していない)のzero_maskを作る。

    まだどの階層のシフトも選んでいないので、全列を「候補あり(全て1)」
    として扱い、パック済み uint64 配列で返す。
    """
    return _pack_masks(np.ones(cols, dtype=bool))

class State:
    """探索処理の状態を保持し、反復 DFS を実行する。

    This class owns the current search path (`key`), the active zero-mask,
    the pruning thresholds, and the best result state seen so far.

    Public API:
    - `search(depth)`: 指定深さまで DFS を実行する
    - `run(depth=None)`: 既定設定を使って探索を実行し、self を返す
    - `max_count`, `results`, `shifts`: 最良結果の集計
    """

    __slots__ = (
        "config",
        "key",
        "primes",
        "params",
        "shift_table",
        "zero_mask",
        "max_depth",
        "target",
        "max_count",
        "shifts",
        "target_shifts",
        "max_shifts",
        "target_results",
        "results",
        "start_time",
        "node_count",
        "pbar",
        "checkpoint_path",
        "checkpoint_interval",
        "_stack",
        "progress_callback",
    )

    def __init__(self, config: SearchConfig | Sequence[int], shift_table: list[NDArray[np.uint64]], max_depth: int | None = None, target: int | None = None, checkpoint_path: str | os.PathLike[str] | None = None, checkpoint_interval: int = 1000, progress_callback=None) -> None:
        # SearchConfig 以外(生の primes 列)が渡された場合は、まず SearchConfig に
        # 正規化してしまう。これにより以降の属性代入を両ケースで共通化でき、
        # max_depth/target の決定ロジックを二重に書かずに済む。
        if not isinstance(config, SearchConfig):
            config = SearchConfig(
                primes=config,
                depth=len(config),
                max_depth=cfg.max_depth if max_depth is None else max_depth,
                target=cfg.target if target is None else target,
                cols=cfg.cols,
            )
        self.config = config
        primes = config.primes
        if len(shift_table) < config.depth:
            raise ValueError(
                f"shift_table の階層数({len(shift_table)})が depth({config.depth})未満です"
            )
        params = config.params
        # if len(params) < config.depth:
        #     raise ValueError(
        #         f"params の階層数({len(params)})が depth({config.depth})未満です"
        #     )
        # params[0] = [1]
        # params[1] = [1]
        # params[2] = [4]
        # params[3] = [4]
        # params[4] = [5,7]
        # params[5] = [1, 11]
        # params[6] = [15]
        # params[7] = [7, 16]
        # params[8] = [9, 13, 17, 21]
        # params[9] = [0, 26]
        # params[10] = [0, 13, 14, 27]
        # params[11] = [2, 4, 9]
        # params[12] = [2, 13, 29]
        # params[13] = [2, 5, 9]  # [1,2,5,9,16,18,26]
        # params[14] = [29]
        # params[15] = [41, 48]
        # params[16] = [8, 32, 33, 45]
        # params[17] = [4, 31]
        # params[18] = [9, 32, 44]
        # params[19] = [16]
        # params[20] = [44]
        for row in range(ROWS):
            params[row] = [i for i in range(row+1,primes[row])]

        self.max_depth = config.max_depth if max_depth is None else max_depth
        self.target = config.target if target is None else target

        self.key: list[int] = []
        self.primes: Sequence[int] = primes
        self.params: Sequence[list[int]] = params
        self.shift_table: list[NDArray[np.uint64]] = shift_table
        self.max_count: int = 0
        self.shifts: list[list[int]] = []
        self.target_shifts: list[list[int]] = []
        self.max_shifts: list[list[int]] = []
        self.target_results: int = 0
        self.results: int = 0
        self.start_time: float = time.time()
        self.node_count: int = 0
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_interval = checkpoint_interval
        self._stack: list[list] = []
        # まだどの階層のシフトも決めていない状態の zero_mask(全列が候補)。
        # チェックポイントから読み込む場合は _load_checkpoint 側で上書きされる。
        self.zero_mask: NDArray[np.uint64] = build_initial_zero_mask(config.cols)
        # GUIなどから進捗を受け取りたい場合に渡すコールバック。
        # dict(node_count, max_count, target_results, depth, key) を引数に呼ばれる。
        self.progress_callback = progress_callback
        self.pbar = tqdm(
            desc="search",
            unit="node",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=self.config.progress_mininterval,
        )

    def count_zero(self, arr: NDArray[np.bool_]) -> int:
        return int(np.sum(np.all(arr == 0, axis=0)))

    def _update_shifts(self) -> None:
        """target 到達パスと最大値パスを順序を保って重複なく公開する。"""
        self.shifts = []
        for path in self.target_shifts + self.max_shifts:
            if path not in self.shifts:
                self.shifts.append(path)

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stack_payload = []
        for level, base_mask, next_idx, next_p in self._stack:
            stack_payload.append({
                "level": int(level),
                # 多倍長整数はJSONの数値としてそのままシリアライズ可能。
                "base_mask": self._mask_to_int(base_mask),
                "next_idx": int(next_idx),
                "next_p": int(next_p),
            })

        saved = {
            "version": 2,
            "settings": {
                "primes": list(self.primes),
                "params": list(self.params),
                "depth": self.config.depth,
                "cols": self.config.cols,
                "max_depth": self.max_depth,
                "target": self.target,
                "traversal_order": "descending",
                "mask_format": "uint64-little-endian",
                "result_format": "target-and-maximum-path-counts",
            },
            "key": list(self.key),
            "zero_mask": self._mask_to_int(self.zero_mask),
            "max_count": self.max_count,
            "results": self.results,
            "target_results": self.target_results,
            "shifts": self.shifts,
            "target_shifts": self.target_shifts,
            "max_shifts": self.max_shifts,
            "node_count": self.node_count,
            "stack": stack_payload,
        }
        # 途中でクラッシュしても既存のチェックポイントを壊さないよう、
        # 一時ファイルに書いてから rename でatomicに置き換える。
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return
        try:
            saved = json.loads(text)
        except json.JSONDecodeError as exc:
            # 旧バージョン(key=value形式のテキスト)のチェックポイントを
            # 読み込もうとした場合に、原因不明のJSONエラーではなく
            # 何が問題かが分かるメッセージにする。
            raise ValueError(
                f"{path} はJSON形式のチェックポイントとして読み込めません。"
                "旧バージョン(key=value形式)のチェックポイントには対応していません。"
                "新しい設定で最初から探索をやり直してください。"
            ) from exc

        if saved.get("version") != 2:
            raise ValueError(f"unsupported checkpoint version: {saved.get('version')}")
        expected_settings = {
            "primes": list(self.primes),
            "params": list(self.params),
            "depth": self.config.depth,
            "cols": self.config.cols,
            "max_depth": self.max_depth,
            "target": self.target,
            "traversal_order": "descending",
            "mask_format": "uint64-little-endian",
            "result_format": "target-and-maximum-path-counts",
        }
        if saved.get("settings") != expected_settings:
            raise ValueError(
                f"{path} の探索設定が現在の設定と一致しません: "
                f"保存値={saved.get('settings')!r}, 現在値={expected_settings!r}"
            )

        self.key = list(saved.get("key", []))
        self.zero_mask = self._int_to_mask(int(saved.get("zero_mask", 0)), self.config.cols)
        self.max_count = int(saved.get("max_count", 0))
        self.results = int(saved.get("results", 0))
        self.target_results = int(saved.get("target_results", 0))
        self.target_shifts = list(saved.get("target_shifts", []))
        self.max_shifts = list(saved.get("max_shifts", []))
        self._update_shifts()
        self.node_count = int(saved.get("node_count", 0))
        raw_stack = saved.get("stack", [])
        self._stack = [
            [
                int(entry["level"]),
                self._int_to_mask(int(entry["base_mask"]), self.config.cols),
                int(entry["next_idx"]),
                int(entry["next_p"]),
            ]
            for entry in raw_stack
        ]

    def report_progress(self, force: bool = False) -> None:
        """
        tqdmの進捗バーを更新する(ログファイルには出さない)。

        呼び出しが多い(再帰の全ノードで呼ばれる)ため、実際の画面再描画は
        tqdm側が mininterval 秒に一度だけに間引いてくれる。force=True の
        ときは set_postfix に refresh=True を渡し、間引かずに必ず再描画する
        (探索開始・終了時など)。
        """
        if self.node_count % self.config.postfix_update_interval == 0:
            self.pbar.update(self.config.postfix_update_interval)
            self.pbar.set_postfix(
                best=self.max_count,
                hits=self.target_results,
                depth=len(self.key),
                key=list(self.key),
                refresh=force,
            )
        if self.checkpoint_path is not None and self.node_count % self.checkpoint_interval == 0:
            self._save_checkpoint()

        if self.progress_callback is not None and (
            force or self.node_count % self.config.postfix_update_interval == 0
        ):
            self.progress_callback(
                {
                    "node_count": self.node_count,
                    "max_count": self.max_count,
                    "target_results": self.target_results,
                    "depth": len(self.key),
                    "key": list(self.key),
                }
            )

    def search(self, depth: int) -> None:
        """
        各階層のシフト値を探索する(反復版・スタックによる明示的DFS)。

        元の実装は「1階層シフトを決める→自分自身を再帰呼び出しして
        次の階層を決める」という再帰関数だったが、depth(ひいては
        再帰の深さ)が大きくなると Python の再帰上限(sys.setrecursionlimit)
        や関数呼び出しオーバーヘッドが問題になりうる。
        ここでは再帰呼び出しの代わりに、階層ごとの「ループの途中状態」を
        自前のスタック `self._stack` に積んで管理することで、同じ探索順序・
        同じ結果を非再帰(反復)で実現する。

        `self._stack` はインスタンス属性として持たせているため、
        `_save_checkpoint`/`_load_checkpoint` からもそのまま読み書きできる。
        そのため、探索の全ノードごとに別のコピーへ同期する必要はなく、
        チェックポイントを実際に保存するタイミング(`checkpoint_interval`
        ノードに1回)でだけ JSON へシリアライズすればよい。

        スタックの各要素は [level, base_mask, next_idx, next_len] の
        4要素からなる可変(list)フレーム:
            level      : このループで値を決める階層(0-indexed)。
                         元の再帰版での level = len(key) - 1 に対応。
            base_mask  : この階層のどの枝を試す場合でも共通して使う
                         「親までの zero_mask」。元の再帰版での
                         prev_mask(= self.zero_mask の呼び出し前の値)
                         に対応する。
            next_idx   : 次に試す `params[level]` 上のインデックス
                         (`range(next_len - 1, -1, -1)`)。実際に使う値は
                         `params[level][next_idx]` であり、primes[level]
                         から作られる `range(p)` ではなく、この
                         `params[level]` の要素(添字)がそのまま
                         shift_table[level] の行インデックスとして使われる。
                         元の再帰版の re-entrant な `for i in range(...)`
                         ループの「途中状態」をこれで表現する。
            next_len   : この階層で試す候補数(= len(params[level]))。
                         next_idx の開始値としても保持している。

        1つの節点(= key の1要素)を「探索し尽くして親に戻る」タイミングは、
        自分の子階層で next_idx が next_len に達した瞬間として検出し、
        そこで元の再帰版の「self.zero_mask = prev_mask; return」に相当する
        後始末(zero_maskの復元・keyのpop)を行う。

        Parameters
        ----------
        depth : int
            探索する階層数(= 使用する素数の個数)。可変。
        """
        if depth == 0:
            return

        key = self.key
        stack = self._stack
        if not stack:
            stack.append([0, self.zero_mask.copy(), len(self.params[0]) - 1, len(self.params[0])])

        while stack:
            frame = stack[-1]
            level, base_mask, next_idx, next_len = frame
            if next_idx < 0:
                finished_base_mask = stack.pop()[1]
                if stack:
                    key.pop()
                    self.zero_mask = stack[-1][1]
                else:
                    self.zero_mask = finished_base_mask  # 最上位まで戻り切った
                continue

            pos = next_idx
            frame[2] = next_idx - 1  # 同じフレームをその場で更新(コピー不要)

            i = self.params[level][pos]  # primes[level]の代わりにparams[level]の値を使う
            key.append(i)
            self.node_count += 1

            # report_progress()(チェックポイント保存を含む)は、
            # key と stack の対応関係が「len(key) == len(stack) - 1」に
            # 揃っているタイミングでしか呼んではいけない。
            # 枝刈り/葉ノード判定/子階層プッシュのどの分岐を取るかが
            # 確定する前(= key.append 直後)に呼んでしまうと、
            # 「i を試している最中」という中途半端な状態のまま
            # チェックポイントに保存されてしまい、再開時に stack 側の
            # next_idx だけが1つ先に進んだ不整合な状態から再開してしまう
            # (key が本来ポップされるべきだったエントリを残したまま
            # 際限なく伸び続けるバグになる)。
            # そのため、分岐の結果が確定した直後(continue する直前、
            # または子階層をpushした直後)に必ず1回だけ呼ぶよう
            # try/finally で保証する。
            try:
                row_complement = self.shift_table[level][i]  # ~row_nonzero(NOT演算済み、事前作成済み)
                node_mask, count = intersect_masks_and_count(base_mask, row_complement)

                if count < self.max_count:
                    key.pop()
                    continue

                if level + 1 >= depth:
                    if count == self.target:
                        message = f"target depth={depth} key={list(key)} count={count}"
                        self.pbar.write(message)
                        logger.info(message)  # ログファイルにも残す(pbar.writeだけだと画面にしか出ない)
                        self.target_results += 1
                        self.target_shifts.append(list(key))

                    if not (depth == self.max_depth and count > self.target):
                        if count > self.max_count:
                            self.max_count = count
                            self.results = 1
                            self.max_shifts = [list(key)]
                        elif count == self.max_count:
                            self.results += 1
                            self.max_shifts.append(list(key))
                    self._update_shifts()

                    key.pop()
                    continue

                self.zero_mask = node_mask
                next_len_child = len(self.params[level + 1])
                stack.append([level + 1, node_mask, next_len_child - 1, next_len_child])
            finally:
                self.report_progress()

    def run(self, depth: int | None = None, resume_from: str | os.PathLike[str] | None = None) -> "State":
        """primes[:depth] を使って深さ depth までの探索を実行するエントリポイント"""
        if depth is not None and depth < 0:
            raise ValueError(f"depth は0以上である必要があります: depth={depth}")
        depth_to_use = self.config.depth if depth is None else depth
        if depth_to_use > len(self.primes):
            raise ValueError(f"depth={depth_to_use} が使用可能な素数の個数({len(self.primes)})を超えています")
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        try:
            self.search(depth_to_use)
            self.report_progress(force=True)
        finally:
            self.pbar.close()
            if self.checkpoint_path is not None:
                self._save_checkpoint()

        return self


class SearchWorker(QThread):
    """探索(State.run())をUIスレッドをブロックせずに実行するためのワーカー。

    重い探索処理をメインスレッドで実行するとGUIが固まってしまうため、
    別スレッド上で実行し、進捗・結果をシグナル経由でメインスレッドへ通知する。
    """

    # 進捗通知: State.report_progress() から渡される辞書をそのまま流す
    progress = Signal(dict)
    # 正常終了: 完了した State インスタンスを渡す
    finished_ok = Signal(object)
    # 異常終了: エラーメッセージ(トレースバック文字列)を渡す
    failed = Signal(str)

    def __init__(self, depth: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.depth = depth

    def run(self) -> None:  # noqa: D401 (QThreadのオーバーライド)
        try:
            config = SearchConfig()
            config.depth = self.depth
            config.__post_init__()  # 入力値の検証(depthの範囲など)を行う

            shift_table = build_shift_table(config.primes[: config.depth], config.cols)
            state = State(
                config,
                shift_table,
                max_depth=config.max_depth,
                target=config.target,
                progress_callback=self.progress.emit,
            )
            state.run(depth=config.depth)
        except Exception:  # noqa: BLE001 (GUIへエラー内容を伝えるため広く捕捉)
            import traceback

            self.failed.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(state)


class MainWindow(QMainWindow):


    def __init__(self):
        super().__init__()
        self.setWindowTitle("反例探索表")

        menubar = QMenuBar()
        file_menu = menubar.addMenu("ファイル")

        exit_action = QAction("終了", self)

        exit_action.triggered.connect(self.close)

        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        self.setMenuBar(menubar)

        self.statusBar().showMessage("準備完了")
        self.status_label = QLabel("行: 0 / 列: 0")
        self.selection_label = QLabel("選択: なし")
        self.zero_count_label = QLabel("ゼロ列数: 0")
        self.statusBar().addPermanentWidget(self.status_label)
        self.statusBar().addPermanentWidget(self.selection_label)
        self.statusBar().addPermanentWidget(self.zero_count_label)

        # --- 中央ウィジェット: 数値入力欄・実行ボタン・進捗バー・結果表示欄 ---
        central = QWidget()
        layout = QVBoxLayout(central)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("探索する深さ(使用する素数の個数):"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, len(PRIMES))
        self.depth_spin.setValue(cfg.depth)
        input_row.addWidget(self.depth_spin)
        input_row.addStretch(1)
        layout.addLayout(input_row)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("探索開始")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_row.addWidget(self.run_button)

        self.clear_button = QPushButton("クリア")
        self.clear_button.clicked.connect(self.on_clear_clicked)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 総ノード数が事前に分からないため不定モードで表示
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.result_edit = QPlainTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setPlaceholderText("ここに探索結果が表示されます。")
        layout.addWidget(self.result_edit)

        self.setCentralWidget(central)

        self._worker: SearchWorker | None = None

    def on_run_clicked(self) -> None:
        """「探索開始」ボタン押下時: state.run() をバックグラウンドで実行する。"""
        if self._worker is not None and self._worker.isRunning():
            return  # 実行中の多重起動を防止

        depth = self.depth_spin.value()

        self.run_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result_edit.clear()
        self.result_edit.appendPlainText(f"深さ {depth} で探索を開始します…")
        self.statusBar().showMessage("探索中…")

        self._worker = SearchWorker(depth, parent=self)
        self._worker.progress.connect(self.on_search_progress)
        self._worker.finished_ok.connect(self.on_search_finished)
        self._worker.failed.connect(self.on_search_failed)
        self._worker.start()

    def on_search_progress(self, info: dict) -> None:
        """探索中の進捗表示を更新する。"""
        self.statusBar().showMessage(
            f"探索中… ノード数={info['node_count']:,} "
            f"最良値={info['max_count']} target到達={info['target_results']} "
            f"深さ={info['depth']}"
        )

    def on_search_finished(self, state: "State") -> None:
        """探索が正常終了したときの表示。"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.statusBar().showMessage("探索完了")

        lines = [
            "探索が完了しました。",
            f"総ノード数: {state.node_count:,}",
            f"最良値(max_count): {state.max_count}",
            f"target({state.target})到達回数: {state.target_results}",
            "",
            "最良値を達成したシフト列:",
        ]
        for shift in state.max_shifts:
            lines.append(f"  {shift}")
        if state.target_shifts:
            lines.append("")
            lines.append(f"target({state.target})を達成したシフト列:")
            for shift in state.target_shifts:
                lines.append(f"  {shift}")

        self.result_edit.setPlainText("\n".join(lines))

    def on_search_failed(self, message: str) -> None:
        """探索中に例外が発生したときの表示。"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.statusBar().showMessage("探索中にエラーが発生しました")
        self.result_edit.setPlainText(f"エラーが発生しました:\n\n{message}")

    def on_clear_clicked(self) -> None:
        """「クリア」ボタン押下時: テキストエリアの内容を消去する。"""
        self.result_edit.clear()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())