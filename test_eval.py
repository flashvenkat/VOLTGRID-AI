"""Local validator for the predictor's documented CSV contract."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "sample_test_input.csv"


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "predictions.csv"
        subprocess.run(
            [sys.executable, str(ROOT / "predict.py"), "--input", str(INPUT), "--output", str(output)],
            check=True,
            cwd=ROOT,
        )
        with INPUT.open(newline="", encoding="utf-8-sig") as source, output.open(
            newline="", encoding="utf-8-sig"
        ) as result:
            input_rows = list(csv.DictReader(source))
            reader = csv.DictReader(result)
            output_rows = list(reader)
            assert reader.fieldnames == ["timestamp", "target_demand_kw"]
            assert len(input_rows) == len(output_rows)
            assert [row["timestamp"] for row in input_rows] == [
                row["timestamp"] for row in output_rows
            ]
            assert all(float(row["target_demand_kw"]) >= 0 for row in output_rows)
    print("SUCCESS")


if __name__ == "__main__":
    main()
