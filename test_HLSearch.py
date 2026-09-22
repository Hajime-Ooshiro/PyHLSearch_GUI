import tempfile
import unittest
from pathlib import Path

import HLSearch


class StateDatabaseTests(unittest.TestCase):
    def make_config(self) -> HLSearch.SearchConfig:
        config = HLSearch.SearchConfig()
        config.primes = [2, 3]
        config.params = [[0, 1], [0, 1, 2]]
        config.depth = 2
        config.max_depth = 3
        config.cols = 16
        config.progress_mininterval = 60.0
        config.postfix_update_interval = 1000
        return config

    def make_state(
        self, database_path: Path | bool
    ) -> HLSearch.State:
        config = self.make_config()
        shift_table = HLSearch.build_shift_table(config.primes, config.cols)
        return HLSearch.State(
            config,
            shift_table,
            db_path=database_path,
            db_batch_size=10,
        )

    def test_fetch_prefix_keys_is_unique_and_orders_by_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(Path(directory) / "results.db")
            try:
                state._record_result(1, 2, [1])
                state._record_result(1, 5, [0])
                state._record_result(1, 2, [1])
                state._flush_db()

                self.assertEqual(state._fetch_prefix_keys(1), [[0], [1]])
                self.assertEqual(state._fetch_prefix_key(1), [0])
                self.assertEqual(state._fetch_prefix_keys(-1), [])
            finally:
                state._close_db()

    def test_delete_db_removes_only_requested_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(Path(directory) / "results.db")
            try:
                state._record_result(2, 1, [0, 0])
                state._record_result(12, 1, [0])
                state._flush_db()
                state._delete_db(12)

                self.assertEqual(state._fetch_prefix_keys(12), [])
                self.assertEqual(state._fetch_prefix_keys(2), [[0, 0]])
            finally:
                state._close_db()

    def test_run_reuses_database_prefix_without_reexploring_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "results.db"
            writer = self.make_state(database_path)
            writer._record_result(1, 1, [0])
            writer._close_db()

            state = self.make_state(database_path)
            try:
                state.run(depth=2)

                self.assertEqual(state.node_count, 3)
                self.assertTrue(state.max_shifts)
                self.assertTrue(all(path[0] == 0 for path in state.max_shifts))
            finally:
                state._close_db()

    def test_database_can_be_disabled(self) -> None:
        state = self.make_state(False)
        try:
            self.assertIsNone(state.db_path)
            self.assertIsNone(state.db_conn)
            self.assertEqual(state._fetch_prefix_keys(1), [])
        finally:
            state._close_db()


if __name__ == "__main__":
    unittest.main()
