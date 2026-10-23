"""Serialization, labels, split ownership and raw-feature normalization contracts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from data import collate_records, encode, encode_input, load_selection
from features.bank import normalize_final
from model import Gemma4FeatureCache
from utils import atomic_json, digest, file_hash

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference/legacy"))
from architecture.data import identity_keys as old_keys
from architecture.patch_when2call import added_candidate as old_candidate
from gemma_decisions.pilot import encode as old_encode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
from common import identity_keys
from when2call_candidates import added_candidate


class DataTests(unittest.TestCase):
    def test_rounded_teacher_probabilities_keep_original_values(self):
        probabilities = [0.0707, 0.2828, 0.2727, 0.2828, 0.0808, 0.0101]
        row = {
            "id": "rounded-teacher",
            "ids": [1] * 6,
            "positions": list(range(6)),
            "target": probabilities,
        }
        batch = collate_records([row])
        self.assertTrue(torch.equal(batch.targets[0], torch.tensor(probabilities)))
        # This is the real rounding boundary that the FP32 sum rejected.
        self.assertGreater(abs(float(batch.targets.sum()) - 1), 1e-4)
        for invalid in (
            [0.2] * 6,
            [-0.1, 0.3, 0.2, 0.2, 0.2, 0.2],
            [float("nan"), 0.2, 0.2, 0.2, 0.2, 0.2],
        ):
            with self.assertRaisesRegex(ValueError, "Invalid target distribution"):
                collate_records([{**row, "target": invalid}])

    def test_serializer_source_and_dataset_repairs_are_unchanged(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        tokenizer = Tokenizer(
            models.BPE(
                vocab={"[UNK]": 0, **{c: i + 1 for i, c in enumerate(alphabet)}},
                merges=[],
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        for state in [
            "Context with punctuation: a = b + 2.",
            {"ticket": "café ☕", "tools": []},
        ]:
            raw = dict(
                id="unicode-é",
                group_id="g",
                state=state,
                question="Choose a value?",
                options=["oui", "non", "日本語"],
                source="fixture",
                kind="choice",
                target=[0.0, 1.0, 0.0],
            )
            for order in [None, [2, 0, 1]]:
                expected = old_encode(raw, tokenizer, 4096, order)
                self.assertEqual(encode(raw, tokenizer, 4096, order), expected)
                unlabelled = {k: v for k, v in raw.items() if k != "target"}
                self.assertEqual(
                    encode_input(unlabelled, tokenizer, 4096, order),
                    {k: v for k, v in expected.items() if k != "target"},
                )
        row = {
            "id": "fixture-é",
            "group_id": "g",
            "state": {"question": "café", "tools": []},
            "question": "décision?",
            "options": ["oui", "non"],
            "source": "fixture",
        }
        self.assertEqual(old_keys(row), identity_keys(row))
        for policy in ["tool_call", "request_for_info", "cannot_answer", "direct"]:
            self.assertEqual(old_candidate(policy, row), added_candidate(policy, row))

    def test_train_and_heldout_groups_must_not_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            atomic_json(
                path,
                {
                    "splits": {
                        "train": [{"id": "1", "group_id": "g"}],
                        "test": [{"id": "2", "group_id": "g"}],
                    }
                },
            )
            with self.assertRaisesRegex(ValueError, "membership overlap"):
                load_selection(path)

    def test_raw_features_are_normalized_once_and_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "train"
            split.mkdir()
            row = {
                "id": "one",
                "ids": [1] * 5,
                "positions": [1, 4],
                "order": [0, 1],
                "target": [1.0, 0.0],
            }
            atomic_json(split / "records.json", [row])
            hidden = torch.randn(5, 16).to(torch.bfloat16)
            norm = {"weight": torch.randn(16), "eps": 1e-6}
            torch.save(norm, root / "final-rmsnorm.pt")
            atomic_json(
                root / "manifest.json",
                {"backbone": {"checkpoint": "fixture"}, "serialization_sha256": "abc"},
            )
            file = "layers/layer-48/shard-00000.pt"
            (split / file).parent.mkdir(parents=True)
            torch.save({"hidden": hidden}, split / file)
            entry = {
                "shard": 0,
                "records_sha256": digest([row]),
                "rows": [{"id": "one", "offset": 0, "tokens": 5}],
                "files": {
                    "layer-48": {
                        "file": file,
                        "sha256": file_hash(split / file),
                        "bytes": (split / file).stat().st_size,
                    }
                },
            }
            (split / "layer-additions.jsonl").write_text(json.dumps(entry) + "\n")
            atomic_json(split / "index.json", {"rows_by_layer": {"layer-48": 1}})
            selection = {
                "identity": {"checkpoint": "fixture", "serialization_sha256": "abc"},
                "splits": {"train": [row]},
            }
            provider = Gemma4FeatureCache(root, selection=selection)
            batch = provider(collate_records([row]))
            self.assertTrue(torch.equal(batch.hidden[0], normalize_final(hidden, norm)))
            changed = {**row, "positions": [2, 4]}
            with self.assertRaisesRegex(ValueError, "input identity"):
                provider(collate_records([changed]))
            selection["identity"]["checkpoint"] = "other"
            with self.assertRaisesRegex(ValueError, "backbone differs"):
                Gemma4FeatureCache(root, selection=selection)


if __name__ == "__main__":
    unittest.main()
