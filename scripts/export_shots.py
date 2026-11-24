"""Write reviewed launch candidates from data/match-labels as scripts.train_shots input.

    python -m scripts.export_shots && python -m scripts.train_shots data/match-labels/shot-labels.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    labels, rows = ROOT / "data/match-labels", []
    for out in sorted(path for path in labels.iterdir() if (path / "launches.json").is_file()):
        verdicts = {}
        for path in out.glob("answers-*.json"):
            verdicts.update({str(int(k)): v for k, v in json.loads(path.read_text()).get("launches", {}).items()})
        for row in json.loads((out / "launches.json").read_text()):
            verdict = verdicts.get(str(row["id"]))
            if verdict and len(row.get("flight", [])) >= 3:
                rows.append({"group": out.name, "label": int(bool(verdict.get("launch"))), "samples": row["flight"]})
    (labels / "shot-labels.json").write_text(json.dumps(rows))
    print(f"{len(rows)} reviewed launches, {sum(r['label'] for r in rows)} confirmed, "
          f"{len({r['group'] for r in rows})} recordings")


if __name__ == "__main__":
    main()
