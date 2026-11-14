import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import src.data
import src.tokenization
from src.utils import load_train_config
from tests import test_data


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks/vietnamese_summarization_data_preparation.ipynb"


class FullDatasetTests(unittest.TestCase):
    def codes(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]

    def test_full_selection_preserves_every_parent_sample_and_group(self):
        splits, parent = test_data.FixedSubsetTests().make_parent()
        original = copy.deepcopy(splits)
        selected, manifest = src.data.select_full_dataset(splits, parent)
        self.assertEqual(manifest["scope"], "full")
        self.assertEqual(src.data.restore_fixed_subsets(splits, parent, manifest), selected)
        for name in splits:
            self.assertEqual(selected[name], sorted(splits[name], key=lambda row: row["id"]))
            self.assertEqual(manifest["splits"][name]["pair_count"], parent["splits"][name]["pair_count"])
            self.assertEqual(manifest["splits"][name]["split_group_ids"], parent["splits"][name]["split_group_ids"])
        self.assertEqual(splits, original)

    def test_preparation_exports_full_splits_and_training_reads_new_manifest(self):
        codes = self.codes()
        splits, parent = test_data.FixedSubsetTests().make_parent()
        config = load_train_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Existing subset files must have no effect on the full-data export.
            old_file = root / "data/processed/train.jsonl"
            old_file.parent.mkdir(parents=True)
            old_file.write_bytes(b"old subset")
            namespace = {
                "src": types.SimpleNamespace(data=src.data), "json": json,
                "cleaned_pairs": [], "cleaning_report": {"removed_counts_by_reason": {}},
                "config_path": ROOT / "configs/train.json", "train_config": config,
                "reference_manifests": {"wikilingua_splits.json": parent}, "drive_root": root,
            }

            def tokenize(**kwargs):
                text = kwargs.get("text", kwargs.get("text_target"))
                return {"input_ids": [1] * (len(text.split()) + 1)}

            module = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenize,
            ))
            with contextlib.redirect_stdout(io.StringIO()), patch.dict("sys.modules", {"transformers": module}):
                with patch.object(src.data, "split_wikilingua_pairs", return_value=(splits, parent)):
                    exec(compile(codes[6], str(NOTEBOOK), "exec"), namespace)
                exec(compile(codes[7], str(NOTEBOOK), "exec"), namespace)
                exec(compile(codes[8], str(NOTEBOOK), "exec"), namespace)
                first_export = copy.deepcopy(namespace["processed_manifest"])
                exec(compile(codes[8], str(NOTEBOOK), "exec"), namespace)
            self.assertEqual(namespace["processed_manifest"], first_export)
            self.assertEqual(old_file.read_bytes(), b"old subset")
            _, loaded, _, fingerprint = src.tokenization.load_tokenization_source(root)
            for name in ["train", "validation"]:
                self.assertEqual(len(loaded[name]), len(splits[name]))
            self.assertEqual(fingerprint, src.data._split_manifest_fingerprint(namespace["data_manifest"]))
            self.assertTrue(all(src.data.verify_prepared_datasets(
                root, first_export, namespace["data_manifest"],
            ).values()))
            report = json.loads((root / config["token_limits_decision"]["selected_report"]).read_text(encoding="utf-8"))
            self.assertEqual(set(report["splits"]), {"train", "validation"})
            self.assertFalse(report["checks"]["test_content_read"])
            # A different saved selection must fail before export or overwrite.
            manifest_path = root / config["paths"]["subset_manifest"]
            changed = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed["scope"] = "different-selection"
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            previous_bytes = manifest_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "differs from the saved selection"):
                exec(compile(codes[8], str(NOTEBOOK), "exec"), namespace)
            self.assertEqual(manifest_path.read_bytes(), previous_bytes)

    def test_preparation_fetches_current_source_and_does_not_restore_small_subset(self):
        codes = self.codes()
        self.assertIn('target_commit = ""', codes[1])
        self.assertIn('["git", "rev-parse", "origin/main"]', codes[1])
        self.assertNotIn("restore_fixed_subsets", "\n".join(codes))
        parent = json.loads((ROOT / "data/manifests/wikilingua_splits.json").read_text(encoding="utf-8"))
        self.assertEqual({name: split["pair_count"] for name, split in parent["splits"].items()}, {
            "train": 15601, "validation": 2034, "test": 1942,
        })
