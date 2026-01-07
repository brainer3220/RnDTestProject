import math
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.xlsx_reader import read_xlsx_table


def summarize_columns(headers, rows):
    stats = {
        h: {"count": 0, "missing": 0, "min": None, "max": None, "mean": 0.0, "m2": 0.0}
        for h in headers
    }

    for row in rows:
        for h, v in zip(headers, row):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                stats[h]["missing"] += 1
                continue
            st = stats[h]
            st["count"] += 1
            if st["min"] is None or v < st["min"]:
                st["min"] = v
            if st["max"] is None or v > st["max"]:
                st["max"] = v
            delta = v - st["mean"]
            st["mean"] += delta / st["count"]
            st["m2"] += delta * (v - st["mean"])

    for st in stats.values():
        if st["count"] > 1:
            st["std"] = math.sqrt(st["m2"] / (st["count"] - 1))
        else:
            st["std"] = None
        st.pop("m2", None)

    return stats


def format_stats(stats, keys):
    lines = []
    for k in keys:
        st = stats.get(k)
        if not st:
            continue
        lines.append(
            f"- `{k}` count={st['count']} missing={st['missing']} min={st['min']:.3f} max={st['max']:.3f} mean={st['mean']:.3f} std={st['std']:.3f}"
        )
    return lines


def main():
    data_dir = Path("data")
    paths = sorted(data_dir.glob("*.xlsx"))

    lines = ["# EDA Report", "", "Generated from raw .xlsx files in `data/`.", ""]

    for path in paths:
        headers, rows = read_xlsx_table(str(path))
        stats = summarize_columns(headers, rows)
        file_size = os.path.getsize(path)
        lines.append(f"## {path.name}")
        lines.append("")
        lines.append(f"- rows: {len(rows)}")
        lines.append(f"- columns: {len(headers)}")
        lines.append(f"- file size: {file_size} bytes")
        lines.append("")
        lines.append("Key columns:")
        lines.extend(format_stats(stats, ["timestamp", "head_x", "head_y", "head_z"]))
        lines.append("")
        missing_cols = [h for h, st in stats.items() if st["missing"] > 0]
        if missing_cols:
            lines.append(f"Missing values detected in {len(missing_cols)} columns (example: {missing_cols[:5]}).")
        else:
            lines.append("No missing values detected.")
        lines.append("")

    output_path = Path("reports/eda_report.md")
    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
