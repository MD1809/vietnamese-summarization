import json
import tempfile
import unittest
from pathlib import Path

from src.train import early_stopping
from src.utils import load_train_config


class EarlyStoppingTests(unittest.TestCase):
    def test_improvement_resets_counter_and_equal_loss_counts_as_failure(self):
        settings = {"patience": 2, "min_delta": 0.0}
        state = None
        flags = []
        for loss in [2.0, 2.0, 1.8, 1.9, 2.1]:
            state, stop = early_stopping(state, loss, settings)
            flags.append(stop)
        self.assertEqual(flags, [False, False, False, False, True])
        self.assertEqual(state, {"best_loss": 1.8, "epochs_without_improvement": 2})

    def test_min_delta_compares_to_last_significant_improvement(self):
        settings = {"patience": 2, "min_delta": 0.1}
        state, _ = early_stopping(None, 2.0, settings)
        state, stop = early_stopping(state, 1.95, settings)
        self.assertFalse(stop)
        self.assertEqual(state["best_loss"], 2.0)
        state, stop = early_stopping(state, 1.85, settings)
        self.assertFalse(stop)
        self.assertEqual(state["epochs_without_improvement"], 0)

    def test_missing_settings_preserve_legacy_behavior(self):
        self.assertEqual(early_stopping(None, 2.0, None), (None, False))

    def test_nonfinite_loss_is_rejected(self):
        for loss in [float("nan"), float("inf"), -1.0]:
            with self.assertRaises(ValueError):
                early_stopping(None, loss, {"patience": 2, "min_delta": 0.0})

    def test_config_rejects_invalid_settings(self):
        config = load_train_config()
        for settings in [None, {}, {"patience": 0, "min_delta": 0.0},
                         {"patience": True, "min_delta": 0.0},
                         {"patience": 2, "min_delta": -0.1},
                         {"patience": 2, "min_delta": float("nan")},
                         {"patience": 2, "min_delta": True}]:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                config["early_stopping"] = settings
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_train_config(path)
