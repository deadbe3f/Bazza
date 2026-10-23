"""Portable roots, environment precedence and the actual launcher arguments."""

import json
import os
import shlex
import shutil
import subprocess
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
            for file in ("config.py", "scripts/run.sh", ".env.example"):
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

    def test_launcher_passes_and_mounts_declared_roots_with_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            capture = root / "docker.json"
            fake = bin_dir / "docker"
            fake.write_text("""#!/usr/bin/env python3
import json,os,sys
if sys.argv[1:3]==['image','inspect']:print('sha256:fixture')
elif sys.argv[1]=='run':
    with open(os.environ['DOCKER_CAPTURE'],'w') as stream:json.dump(sys.argv[1:],stream)
else:raise SystemExit('unexpected docker invocation')
""")
            fake.chmod(0o755)
            config = root / "settings.py"
            config.write_text(
                "# Launch-argument fixture; fake Docker does not execute it.\n"
            )
            paths = {
                name: root / name.lower()
                for name in VARIABLES
                if name not in ("OVERFIT_URL", "BAZZA_SELECTION_PATH")
            }
            for path in paths.values():
                path.mkdir()
            selection = root / "selection files/selection.json"
            selection.parent.mkdir()
            selection.write_text("{}")
            env_file = root / "local.env"
            env_file.write_text(
                "\n".join(
                    f"{name}={shlex.quote(str(path))}" for name, path in paths.items()
                )
                + f"\nBAZZA_SELECTION_PATH={shlex.quote(str(selection))}\nOVERFIT_URL=http://localhost:1234\n"
            )
            cache = root / "CLI cache with spaces"
            cache.mkdir()
            missing = root / "missing input root"
            output = root / "new output"
            env = {
                **clean_environment(),
                "BAZZA_DATA_ROOT": str(root / "inherited data"),
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "DOCKER_CAPTURE": str(capture),
            }
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/run.sh"),
                    "train",
                    "--config",
                    str(config),
                    "--env-file",
                    str(env_file),
                    "--cache-root",
                    str(cache),
                    "--data-root",
                    str(missing),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(capture.read_text())
            forwarded = [args[i + 1] for i, arg in enumerate(args) if arg == "-e"]
            self.assertIn("BAZZA_CACHE_ROOT=" + str(cache), forwarded)
            self.assertIn("BAZZA_DATA_ROOT=" + str(missing), forwarded)
            self.assertIn("BAZZA_SELECTION_PATH=" + str(selection), forwarded)
            self.assertIn("OVERFIT_URL=http://localhost:1234", forwarded)
            mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "-v"]
            self.assertIn(f"{cache}:{cache}:ro", mounts)
            self.assertIn(f"{output}:{output}:rw", mounts)
            self.assertEqual(
                [entry for entry in mounts if entry.endswith(":rw")],
                [f"{output}:{output}:rw"],
            )
            self.assertEqual(len(mounts), len(set(mounts)))
            self.assertNotIn("--gpus", args)
            self.assertFalse(missing.exists())
            self.assertTrue(output.is_dir())


if __name__ == "__main__":
    unittest.main()
