"""Interchangeable backbones and explicit Python/HF config boundaries."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from configuration import load_config
from data import BatchLoader, collate_records
from losses import DecisionLoss
from model import Bazza, BazzaDecisionHead, Gemma4, Gemma4FeatureCache
from settings import Config, export_to_hf_config
from trainer import Trainer
from utils import atomic_json, file_hash


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, ids, mask):
        return self.dropout(self.embedding(ids))


class BackboneTests(unittest.TestCase):
    def test_live_and_cache_share_model_forward_and_head_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        live = Gemma4(encoder=TinyEncoder())
        rows = [
            dict(
                id=str(i),
                cache=f"{i}.pt",
                ids=[1, 2, 3, 4] + [5] * i,
                positions=[1, 3 + i],
                target=[1.0, 0.0],
            )
            for i in range(2)
        ]
        batch = collate_records(rows)
        with torch.no_grad():
            hidden = live(batch).hidden
        letter = torch.randn(26, 16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for i, row in enumerate(rows):
                torch.save(
                    {"hidden": hidden[i, : len(row["ids"])]}, root / row["cache"]
                )
                files[row["id"]] = {
                    "file": row["cache"],
                    "sha256": file_hash(root / row["cache"]),
                }
            atomic_json(root / "index.json", {"files": files})
            with patch(
                "model.gemma_backbone.load_gemma",
                side_effect=AssertionError("Cache must not load Gemma weights"),
            ):
                cached = Gemma4FeatureCache(
                    root,
                    selection={"identity": {}, "splits": {"train": rows}},
                    format="pt",
                )
                self.assertEqual(list(cached.parameters()), [])
                foreign = cached.prepare_batch(rows, "train")
                foreign.feature_source = "other-cache"
                with self.assertRaisesRegex(
                    ValueError, "different feature space or cache"
                ):
                    cached(foreign)
                models = []
                for backbone in (live, cached):
                    torch.manual_seed(7)
                    head = BazzaDecisionHead(
                        letter,
                        30,
                        encoder_width=16,
                        width=8,
                        attention_heads=2,
                        low_steps=1,
                        high_cycles=1,
                        gate_width=4,
                        rank=2,
                    )
                    model = Bazza(backbone, head).train()
                    models.append(model)
                    self.assertFalse(model.backbone.training)
                    prepared = BatchLoader(backbone.prepare_batch, "cpu", 2)(
                        rows, "train"
                    )
                    output = model(prepared).logits
                    loss, _ = DecisionLoss()(output, prepared)
                    loss.backward()
                    if len(models) == 1:
                        expected = output.detach()
                    else:
                        self.assertTrue(torch.equal(expected, output))
                for (name, left), (_, right) in zip(
                    models[0].head.named_parameters(), models[1].head.named_parameters()
                ):
                    self.assertEqual(left.grad is None, right.grad is None, name)
                    if left.grad is not None:
                        self.assertTrue(torch.equal(left.grad, right.grad), name)
                self.assertTrue(all(p.grad is None for p in live.encoder.parameters()))
                model = models[0]
                optimizer = torch.optim.AdamW(model.head.parameters())
                saved = Trainer(model, optimizer, device="cpu").state_dict()
                self.assertFalse(any(k.startswith("backbone.") for k in saved["head"]))
                self.assertEqual(list(saved["head"]), list(model.head.state_dict()))

    def test_prefix_or_extra_taps_are_not_silently_interpreted_as_final_states(self):
        with self.assertRaisesRegex(ValueError, "prefix/tap fusion"):
            Gemma4FeatureCache("/unused", selection={}, layers=(24, 48))


class ConfigTests(unittest.TestCase):
    def test_python_values_paths_and_export_have_separate_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "experiment.py"
            path.write_text(
                "from pathlib import Path\nfrom settings import Config\nconfig=Config(data_selection=Path('selection.json'),feature_format='pt',feature_cache_path=Path('cache'),learning_rate=2e-5,batch_size=16*2)\n"
            )
            config = load_config(path)
            self.assertEqual(config.learning_rate, 2e-5)
            self.assertEqual(config.batch_size, 32)
            self.assertEqual(config.data_selection, root / "selection.json")
            self.assertEqual(config.feature_cache_path, root / "cache")
            output = export_to_hf_config(config, root / "export/config.json")
            self.assertEqual(
                output, json.loads((root / "export/config.json").read_text())
            )
            self.assertEqual(output["model_type"], "bazza")
            self.assertNotIn(str(root), json.dumps(output))
            self.assertNotIn("bazza_training", output)
            self.assertNotIn("auto_map", output)
            changed = replace(config, learning_rate=1e-4)
            self.assertEqual(output, export_to_hf_config(changed))
            self.assertEqual(
                export_to_hf_config(config, include_training=True)["bazza_training"][
                    "learning_rate"
                ],
                2e-5,
            )
            self.assertEqual(config.learning_rate, 2e-5)
            with self.assertRaisesRegex(ValueError, "Python file"):
                load_config(root / "old.json")
            path.write_text("config = {}\n")
            with self.assertRaisesRegex(TypeError, "Config"):
                load_config(path)


class PrecisionTests(unittest.TestCase):
    def test_fp32_head_accepts_real_bf16_cache_without_altering_cached_states(self):
        from configuration import make_model

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "id": "a",
                "cache": "a.pt",
                "ids": [1, 2, 3, 4],
                "positions": [1, 3],
                "target": [1.0, 0.0],
            }
            hidden = torch.randn(4, 16).to(torch.bfloat16)
            torch.save({"hidden": hidden}, root / "a.pt")
            torch.save(
                {"weight": torch.randn(26, 16).to(torch.bfloat16), "softcap": 30.0},
                root / "letters.pt",
            )
            config = Config(
                data_selection=root / "selection.json",
                feature_format="pt",
                feature_cache_path=root,
                letter_readout_path=root / "letters.pt",
                encoder_width=16,
                hidden_width=8,
                attention_heads=2,
                low_steps=1,
                high_cycles=1,
                gate_width=4,
                adapter_rank=2,
                precision="fp32",
            )
            selection = {"identity": {}, "splits": {"train": [row]}}
            model, _ = make_model(config, selection)
            batch = model.backbone.prepare_batch([row], "train")
            output = model(batch).logits
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output.dtype, torch.float32)
            self.assertEqual(model.head.letter.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(hidden, batch.hidden[0]))


if __name__ == "__main__":
    unittest.main()
