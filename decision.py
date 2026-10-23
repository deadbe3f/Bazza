"""Load a trained Bazza model and make a System 1 decision on a new request."""

import json
import os

from systemone import SystemOne

model = SystemOne.from_checkpoint(
    os.environ["BAZZA_CHECKPOINT"],
    backbone_path=os.environ["BAZZA_BACKBONE_PATH"],
    device="cuda",
)

result = model(
    state={
        "ticket": "Order #1182 arrived broken. The customer requests a refund.",
        "order_age_days": 40,
        "refund_policy_days": 30,
    },
    questions={
        "refund": {
            "type": "noul",
            "instructions": "Is the refund allowed under the policy?",
            "criteria": {
                "true": "The refund is allowed by the stated policy.",
                "false": "The refund is not allowed by the stated policy.",
            },
        },
        "action": {
            "type": "choice",
            "instructions": "Select the next action using the stated policy.",
            "criteria": {
                "refund_order": "Issue a refund because the policy permits it.",
                "explain_policy": "Explain that the order is outside the refund window.",
                "ask_order_age": "Ask the order age because it is unknown.",
            },
        },
    },
)

print(json.dumps(result, indent=2))
print("Refund probability:", result["answers"]["refund"]["noul"])
print("Selected action:", result["answers"]["action"]["choice"])
