"""Portable configuration roots and environment overrides."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configuration import load_config
from settings import export_to_hf_config

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = (
    "BAZZA_ROOT",
    "BAZZA_PROJECT_ROOT",
    "BAZZA_DATA_ROOT",
    "BAZZA_CACHE_ROOT",
    "BAZZA_MODEL_ROOT",
    "BAZZA_RUN_ROOT",
    "BAZZA_SELECTION_PATH",
    "OVERFIT_URL",
)


def clean_environment():
    return {key: value for key, value in os.environ.items() if key not in VARIABLES}


class PortablePathsTests(unittest.TestCase):
    def test_default_config_can_move_to_another_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "another checkout"
            clone.mkdir()
            shutil.copy2(ROOT / "config.py", clone / "config.py")
            with patch.dict(os.environ, clean_environment(), clear=True):
                config = load_config(clone / "config.py")
            self.assertEqual(config.root, clone / "build")
            self.assertEqual(config.project_root, clone.parent)
            self.assertEqual(config.data_root, clone / "build/data")
            self.assertEqual(
                config.data_selection,
                clone / "build/data/groundbeef/v0.5/selection.json",
            )
            self.assertEqual(config.cache_root, clone / "build/features/bazza")
            self.assertEqual(
                config.backbone_path, clone / "build/weights/Gemma4-12B-it"
            )
            self.assertEqual(config.overfit_url, "")
            for file in ("config.py", ".env.example"):
                self.assertNotIn("/media/adrian", (ROOT / file).read_text())

    def test_environment_overrides_derive_settings_and_stay_out_of_hf_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                name: str(root / name.lower())
                for name in VARIABLES
                if name != "OVERFIT_URL"
            }
            values["OVERFIT_URL"] = "http://localhost:1234"
            with patch.dict(os.environ, {**clean_environment(), **values}, clear=True):
                config = load_config(ROOT / "config.py")
            self.assertEqual(
                config.data_selection, Path(values["BAZZA_SELECTION_PATH"])
            )
            self.assertEqual(
                config.feature_banks["old"],
                Path(values["BAZZA_CACHE_ROOT"])
                / "groundbeef-v0.4/gemma4-12b-it-nf4-layer-taps",
            )
            self.assertNotIn("kiln_model_path", config.to_dict())
            self.assertEqual(config.overfit_url, "http://localhost:1234")
            self.assertNotIn(
                str(root),
                json.dumps(export_to_hf_config(config, include_training=True)),
            )
            self.assertEqual(config.to_dict()["data_root"], values["BAZZA_DATA_ROOT"])


if __name__ == "__main__":
    unittest.main()
