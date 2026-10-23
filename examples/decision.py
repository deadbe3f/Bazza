"""Load a trained Bazza model and make a System 1 decision on a new request."""

import json
import os
import sys
from pathlib import Path

# Permit `python examples/decision.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from systemone import SystemOne

model = SystemOne.from_checkpoint(
    os.environ["BAZZA_CHECKPOINT"],
    backbone_path=os.environ["BAZZA_BACKBONE_PATH"],
    device="cuda",
)
request = json.loads(Path(__file__).with_name("decision.json").read_text())
result = model(state=request["state"], questions=request["questions"])
print(json.dumps(result, indent=2))
print("Refund probability:", result["answers"]["refund"]["noul"])
print("Selected action:", result["answers"]["action"]["choice"])
