"""Export the 344 supplied structure annotations as a compact, versionable CSV."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microc.data import annotations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/structures.csv"))
    args = ap.parse_args()
    table = annotations(args.workbook)
    if len(table) != 344 or table.type.value_counts().to_dict() != {"CHIN": 250, "OPCID": 68, "CHID": 26}:
        raise ValueError("The supplied annotation workbook does not contain the expected 344 structures")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} annotations to {args.out}")


if __name__ == "__main__": main()
