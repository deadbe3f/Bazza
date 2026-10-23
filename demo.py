"""Inspect one fixed example through the same model/input path as evaluation."""
import json
from evaluate import parser, run


def main():
    cli = parser()
    cli.description = __doc__
    cli.add_argument("--example-id", required=True)
    args = cli.parse_args()
    report = run(args)
    if report is not None:
        entry = json.loads((args.output_dir / "predictions.jsonl").read_text())["predictions"][0]
        chosen = max(range(len(entry["logits"])), key=entry["logits"].__getitem__)
        print(json.dumps({"id": entry["id"], "answer_index": chosen, "letter": chr(65 + chosen)}))


if __name__ == "__main__":
    main()
