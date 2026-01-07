import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assessment import smooth_rows
from src.xlsx_reader import read_xlsx_table


def _series_for(headers, rows, joint, axis):
    key = f"{joint}_{axis}"
    if key not in headers:
        raise ValueError(f"Missing column: {key}")
    idx = headers.index(key)
    values = []
    for row in rows:
        v = row[idx]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            values.append(None)
        else:
            values.append(float(v))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .xlsx file")
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"],
        help="Joint base names (without _x/_y/_z)",
    )
    parser.add_argument("--axis", default="x", choices=["x", "y", "z"], help="Axis to plot")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install with: uv pip install matplotlib")
        sys.exit(1)

    headers, rows = read_xlsx_table(args.file)
    smoothed = smooth_rows(headers, rows, window=5, alpha=0.2)

    fig, axes = plt.subplots(len(args.joints), 1, figsize=(10, 2.5 * len(args.joints)), sharex=True)
    if len(args.joints) == 1:
        axes = [axes]

    for ax, joint in zip(axes, args.joints):
        raw = _series_for(headers, rows, joint, args.axis)
        filt = _series_for(headers, smoothed, joint, args.axis)
        ax.plot(raw, label="raw", alpha=0.6, linewidth=1.0)
        ax.plot(filt, label="filtered", alpha=0.9, linewidth=1.2)
        ax.set_title(f"{joint}_{args.axis}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    output = args.output or f"reports/qa_{Path(args.file).stem}_{args.axis}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
