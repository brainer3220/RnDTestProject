#!/usr/bin/env python3
"""
NASM Overhead Squat Assessment Runner

This script processes motion capture data files and generates a comprehensive
NASM overhead squat assessment report for each subject.

Usage:
    python run_assessment.py [--use-isolation-forest] [--output-json]
"""

from pathlib import Path
import argparse
import json
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assessment import (
    assess_frames,
    maybe_filter_outliers,
    smooth_rows,
    NASMAssessmentResult,
    KneeDeviation,
    TorsoDeviation,
    LumbarDeviation,
)
from src.xlsx_reader import read_xlsx_table


def format_assessment_report(result: NASMAssessmentResult) -> list[str]:
    """Format a single subject's assessment as markdown lines."""
    lines = []
    
    lines.append(f"## Subject {result.subject_id}")
    lines.append("")
    
    # Summary statistics
    lines.append("### Summary")
    lines.append(f"- **Total frames**: {result.total_frames}")
    lines.append(f"- **Valid frames analyzed**: {result.valid_frames}")
    lines.append(f"- **Squat repetitions detected**: {len(result.squat_phases)}")
    
    if result.squat_phases:
        depths = [p.max_depth for p in result.squat_phases]
        lines.append(f"- **Average squat depth**: {sum(depths)/len(depths)*100:.1f}%")
    lines.append("")
    
    # Compensation patterns detected
    if result.compensation_flags:
        lines.append("### ⚠️ Compensation Patterns Detected")
        for flag in result.compensation_flags:
            lines.append(f"- {flag}")
        lines.append("")
    
    # Anterior View - Knee Assessment
    lines.append("### Anterior View: Knee Assessment")
    lines.append("")
    lines.append("| Metric | Left Knee | Right Knee |")
    lines.append("|--------|-----------|------------|")
    
    lk = result.left_knee_valgus
    rk = result.right_knee_valgus
    
    lines.append(f"| **Classification** | {result.left_knee_classification.value} | {result.right_knee_classification.value} |")
    lines.append(f"| Mean valgus angle (°) | {lk.mean:+.2f} | {rk.mean:+.2f} |")
    lines.append(f"| Std deviation (°) | {lk.std:.2f} | {rk.std:.2f} |")
    lines.append(f"| Range (°) | [{lk.min:+.2f}, {lk.max:+.2f}] | [{rk.min:+.2f}, {rk.max:+.2f}] |")
    lines.append(f"| 5th percentile (°) | {lk.p5:+.2f} | {rk.p5:+.2f} |")
    lines.append(f"| 95th percentile (°) | {lk.p95:+.2f} | {rk.p95:+.2f} |")
    lines.append("")
    
    # Direction counts
    left_valgus_frames = sum(1 for f in result.frame_metrics 
                            if f.left_knee_deviation == KneeDeviation.VALGUS)
    left_varus_frames = sum(1 for f in result.frame_metrics 
                           if f.left_knee_deviation == KneeDeviation.VARUS)
    right_valgus_frames = sum(1 for f in result.frame_metrics 
                             if f.right_knee_deviation == KneeDeviation.VALGUS)
    right_varus_frames = sum(1 for f in result.frame_metrics 
                            if f.right_knee_deviation == KneeDeviation.VARUS)
    
    lines.append("**Frame-by-frame deviation counts:**")
    lines.append(f"- Left knee: {left_valgus_frames} valgus (inward), {left_varus_frames} varus (outward)")
    lines.append(f"- Right knee: {right_valgus_frames} valgus (inward), {right_varus_frames} varus (outward)")
    lines.append("")
    
    # Lateral View - Torso & Lumbar Assessment  
    lines.append("### Lateral View: Torso & Lumbar Assessment")
    lines.append("")
    
    tl = result.torso_lean
    ld = result.lumbar_deviation
    
    lines.append("| Metric | Torso Lean | Lumbar Angle |")
    lines.append("|--------|------------|--------------|")
    lines.append(f"| **Classification** | {result.torso_classification.value} | {result.lumbar_classification.value} |")
    lines.append(f"| Mean (°) | {tl.mean:+.2f} | {ld.mean:.2f} |")
    lines.append(f"| Std deviation (°) | {tl.std:.2f} | {ld.std:.2f} |")
    lines.append(f"| Range (°) | [{tl.min:+.2f}, {tl.max:+.2f}] | [{ld.min:.2f}, {ld.max:.2f}] |")
    lines.append(f"| 5th percentile (°) | {tl.p5:+.2f} | {ld.p5:.2f} |")
    lines.append(f"| 95th percentile (°) | {tl.p95:+.2f} | {ld.p95:.2f} |")
    lines.append("")
    
    # Clinical interpretation
    lines.append("### Clinical Interpretation")
    lines.append("")
    
    # Knee interpretation
    if result.left_knee_classification == KneeDeviation.VALGUS or \
       result.right_knee_classification == KneeDeviation.VALGUS:
        lines.append("**Knee Valgus (Inward):** Possible weak gluteus medius/maximus, ")
        lines.append("tight adductors, or ankle dorsiflexion limitation.")
        lines.append("")
    
    if result.left_knee_classification == KneeDeviation.VARUS or \
       result.right_knee_classification == KneeDeviation.VARUS:
        lines.append("**Knee Varus (Outward):** Possible tight lateral structures, ")
        lines.append("weak medial knee stabilizers, or hip external rotation dominance.")
        lines.append("")
    
    # Torso interpretation
    if result.torso_classification == TorsoDeviation.EXCESSIVE_FORWARD:
        lines.append("**Excessive Forward Lean:** Possible tight hip flexors/soleus, ")
        lines.append("weak core/erector spinae, or limited ankle dorsiflexion.")
        lines.append("")
    
    # Lumbar interpretation
    if result.lumbar_classification == LumbarDeviation.EXCESSIVE_LORDOSIS:
        lines.append("**Low Back Arches (Hyperlordosis):** Possible tight hip flexors, ")
        lines.append("weak abdominals/gluteals, or anterior pelvic tilt.")
        lines.append("")
    
    if result.lumbar_classification == LumbarDeviation.FLEXION:
        lines.append("**Low Back Flexion:** Possible tight hamstrings, ")
        lines.append("weak erector spinae, or posterior pelvic tilt.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return lines


def result_to_dict(result: NASMAssessmentResult) -> dict:
    """Convert assessment result to JSON-serializable dictionary."""
    return {
        "subject_id": result.subject_id,
        "total_frames": result.total_frames,
        "valid_frames": result.valid_frames,
        "squat_phases": [
            {
                "start_frame": p.start_frame,
                "bottom_frame": p.bottom_frame,
                "end_frame": p.end_frame,
                "max_depth": p.max_depth,
            }
            for p in result.squat_phases
        ],
        "left_knee_valgus": {
            "mean": result.left_knee_valgus.mean,
            "std": result.left_knee_valgus.std,
            "min": result.left_knee_valgus.min,
            "max": result.left_knee_valgus.max,
            "p5": result.left_knee_valgus.p5,
            "p95": result.left_knee_valgus.p95,
        },
        "right_knee_valgus": {
            "mean": result.right_knee_valgus.mean,
            "std": result.right_knee_valgus.std,
            "min": result.right_knee_valgus.min,
            "max": result.right_knee_valgus.max,
            "p5": result.right_knee_valgus.p5,
            "p95": result.right_knee_valgus.p95,
        },
        "torso_lean": {
            "mean": result.torso_lean.mean,
            "std": result.torso_lean.std,
            "min": result.torso_lean.min,
            "max": result.torso_lean.max,
            "p5": result.torso_lean.p5,
            "p95": result.torso_lean.p95,
        },
        "lumbar_deviation": {
            "mean": result.lumbar_deviation.mean,
            "std": result.lumbar_deviation.std,
            "min": result.lumbar_deviation.min,
            "max": result.lumbar_deviation.max,
            "p5": result.lumbar_deviation.p5,
            "p95": result.lumbar_deviation.p95,
        },
        "classifications": {
            "left_knee": result.left_knee_classification.value,
            "right_knee": result.right_knee_classification.value,
            "torso": result.torso_classification.value,
            "lumbar": result.lumbar_classification.value,
        },
        "compensation_flags": result.compensation_flags,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NASM Overhead Squat Assessment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-isolation-forest",
        action="store_true",
        help="Use IsolationForest for velocity-based outlier detection.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Also output results as JSON for programmatic access.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Median filter window size (default: 5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor (default: 0.2).",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    output_md = Path("reports/results_report.md")
    output_json = Path("reports/results_report.json")
    
    # Report header
    lines = [
        "# NASM Overhead Squat Assessment Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Assessment Criteria",
        "",
        "This report evaluates subjects based on the NASM Overhead Squat Assessment protocol:",
        "",
        "| View | Checkpoints | Compensation Signs |",
        "|------|-------------|-------------------|",
        "| Anterior | Knees | Move inward (valgus) or outward (varus) |",
        "| Lateral | Low back | Excessive arch (hyperlordosis) |",
        "| Lateral | Torso | Excessive forward lean |",
        "",
        "**Angle Convention:**",
        "- Knee valgus angle: Positive = inward (valgus), Negative = outward (varus)",
        "- Torso lean: Positive = forward, Negative = backward",
        "- Lumbar angle: 160-180° = neutral, <150° = hyperlordosis",
        "",
        "---",
        "",
    ]
    
    all_results = []
    
    for path in sorted(data_dir.glob("*.xlsx")):
        print(f"Processing {path.name}...")
        
        # Read data
        headers, rows = read_xlsx_table(str(path))
        
        # Optional: outlier filtering
        if args.use_isolation_forest:
            rows = maybe_filter_outliers(
                headers,
                rows,
                joints=[
                    "l_hip", "r_hip", "l_knee", "r_knee",
                    "l_ankle", "r_ankle", "waist", "torso",
                    "l_shoulder", "r_shoulder",
                ],
            )
        
        # Smooth data
        smoothed = smooth_rows(headers, rows, window=args.window, alpha=args.alpha)
        
        # Run assessment
        result = assess_frames(headers, smoothed, subject_id=path.stem)
        all_results.append(result)
        
        # Format report
        lines.extend(format_assessment_report(result))
    
    # Write markdown report
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines))
    print(f"✓ Wrote markdown report: {output_md}")
    
    # Write JSON report
    if args.output_json:
        json_data = {
            "generated": datetime.now().isoformat(),
            "subjects": [result_to_dict(r) for r in all_results],
        }
        output_json.write_text(json.dumps(json_data, indent=2))
        print(f"✓ Wrote JSON report: {output_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("ASSESSMENT SUMMARY")
    print("="*60)
    for result in all_results:
        flags = ", ".join(result.compensation_flags) if result.compensation_flags else "None"
        print(f"\nSubject {result.subject_id}:")
        print(f"  Left knee:  {result.left_knee_classification.value:10s} (mean: {result.left_knee_valgus.mean:+.1f}°)")
        print(f"  Right knee: {result.right_knee_classification.value:10s} (mean: {result.right_knee_valgus.mean:+.1f}°)")
        print(f"  Torso lean: {result.torso_classification.value:10s} (mean: {result.torso_lean.mean:+.1f}°)")
        print(f"  Lumbar:     {result.lumbar_classification.value:10s} (mean: {result.lumbar_deviation.mean:.1f}°)")
        print(f"  Compensations: {flags}")


if __name__ == "__main__":
    main()
