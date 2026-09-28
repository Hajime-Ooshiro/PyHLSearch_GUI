import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import HLSearch


class StateDatabaseTests(unittest.TestCase):
    def make_config(self) -> HLSearch.SearchConfig:
        return HLSearch.SearchConfig(
            primes=[2, 3],
            params=[[0, 1], [0, 1, 2]],
            depth=2,
            max_depth=3,
            cols=16,
            progress_mininterval=60.0,
            postfix_update_interval=1000,
        )

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
            progress_factory=None,
        )

    def test_fetch_prefix_keys_is_unique_and_orders_by_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(Path(directory) / "results.db")
            try:
                database = state.database
                self.assertIsNotNone(database)
                database.record_result(1, 2, 1, [1])
                database.record_result(1, 5, 1, [0])
                database.record_result(1, 2, 2, [1])
                database.flush()

                self.assertEqual(database.fetch_prefix_keys(1), [[0], [1]])
                self.assertEqual(database.fetch_prefix_key(1), [0])
                self.assertEqual(database.fetch_prefix_keys(-1), [])
            finally:
                if state.database is not None:
                    state.database.close()

    def test_delete_db_removes_only_requested_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(Path(directory) / "results.db")
            try:
                database = state.database
                self.assertIsNotNone(database)
                database.record_result(2, 1, 1, [0, 0])
                database.record_result(12, 1, 1, [0])
                database.flush()
                database.delete_results(12)

                self.assertEqual(database.fetch_prefix_keys(12), [])
                self.assertEqual(database.fetch_prefix_keys(2), [[0, 0]])
            finally:
                if state.database is not None:
                    state.database.close()

    def test_run_reuses_database_prefix_without_reexploring_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "results.db"
            writer = self.make_state(database_path)
            self.assertIsNotNone(writer.database)
            writer.database.record_result(1, 1, 1, [0])
            writer.database.close()

            state = self.make_state(database_path)
            try:
                state.run(depth=2)

                self.assertEqual(state.node_count, 3)
                self.assertTrue(state.max_shifts)
                self.assertTrue(all(path[0] == 0 for path in state.max_shifts))
            finally:
                if state.database is not None:
                    state.database.close()

    def test_run_saves_its_maximum_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(Path(directory) / "results.db")
            try:
                state.run(depth=2)

                self.assertIsNotNone(state.database)
                self.assertEqual(
                    state.database.fetch_prefix_keys(2), state.max_shifts
                )
            finally:
                if state.database is not None:
                    state.database.close()

    def test_run_updates_results_at_max_depth(self) -> None:
        config = self.make_config()
        config.max_depth = config.depth
        shift_table = HLSearch.build_shift_table(config.primes, config.cols)
        state = HLSearch.State(
            config, shift_table, db_path=False, progress_factory=None
        )
        state.run()

        self.assertGreater(state.max_count, 0)
        self.assertGreater(state.results, 0)
        self.assertTrue(state.max_shifts)

    def test_database_can_be_disabled(self) -> None:
        state = self.make_state(False)
        try:
            self.assertIsNone(state.database)
        finally:
            if state.database is not None:
                state.database.close()

    def test_show_progress_runs_console_search_at_requested_depth(self) -> None:
        with patch("HLSearch.run_console_search") as run_console_search:
            exit_code = HLSearch.main(["--show-progress", "--depth", "2"])

        self.assertEqual(exit_code, 0)
        run_console_search.assert_called_once_with(2)

    def test_depth_is_passed_to_gui(self) -> None:
        app = MagicMock()
        app.exec.return_value = 0
        with (
            patch("HLSearch.QApplication", return_value=app),
            patch("HLSearch.MainWindow") as main_window,
        ):
            exit_code = HLSearch.main(["--depth", "2"])

        self.assertEqual(exit_code, 0)
        main_window.assert_called_once_with(2)

    def test_injected_progress_reporter_receives_updates(self) -> None:
        class ProgressSpy:
            def __init__(self) -> None:
                self.updates: list[int] = []
                self.closed = False

            def update(self, n: int = 1) -> None:
                self.updates.append(n)

            def set_postfix(self, **kwargs: object) -> None:
                pass

            def close(self) -> None:
                self.closed = True

        config = self.make_config()
        config.postfix_update_interval = 1
        reporter = ProgressSpy()
        shift_table = HLSearch.build_shift_table(config.primes, config.cols)
        state = HLSearch.State(
            config,
            shift_table,
            db_path=False,
            progress_factory=lambda **kwargs: reporter,
        )
        state.run()

        self.assertTrue(reporter.updates)
        self.assertTrue(reporter.closed)


class SearchConfigTests(unittest.TestCase):
    def test_validation_runs_when_constructed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cols は正の整数"):
            HLSearch.SearchConfig(cols=0)

    def test_mutable_defaults_are_not_shared(self) -> None:
        first = HLSearch.SearchConfig()
        second = HLSearch.SearchConfig()

        first.primes[0] = -1
        first.params[0].append(99)

        self.assertEqual(second.primes[0], 2)
        self.assertNotIn(99, second.params[0])


if __name__ == "__main__":
    unittest.main()
