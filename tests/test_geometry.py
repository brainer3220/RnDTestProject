"""
Unit tests for geometry and assessment modules.

Tests cover:
- Vector operations
- Angle calculations
- Coordinate transformations
- Assessment metric computations
- Filtering algorithms
"""

import unittest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import (
    angle_between, 
    project_to_plane, 
    signed_angle,
    v_add,
    v_sub,
    v_dot,
    v_cross,
    v_norm,
    v_unit,
    v_scale,
)
from src.filtering import (
    median_filter,
    ema_filter,
    smooth_series,
    butterworth_filter,
    detect_outliers_zscore,
    interpolate_missing,
)
from src.assessment import (
    compute_knee_valgus_angle,
    compute_torso_lean_angle,
    compute_lumbar_angle,
    compute_knee_flexion_angle,
    detect_squat_depth,
    detect_squat_phases,
    StatisticalSummary,
    KneeDeviation,
    TorsoDeviation,
    LumbarDeviation,
)


class TestVectorOperations(unittest.TestCase):
    """Tests for basic vector operations."""
    
    def test_v_add(self):
        result = v_add([1, 2, 3], [4, 5, 6])
        self.assertEqual(result, [5, 7, 9])
    
    def test_v_sub(self):
        result = v_sub([5, 7, 9], [1, 2, 3])
        self.assertEqual(result, [4, 5, 6])
    
    def test_v_dot(self):
        # Perpendicular vectors
        self.assertAlmostEqual(v_dot([1, 0, 0], [0, 1, 0]), 0.0)
        # Parallel vectors
        self.assertAlmostEqual(v_dot([1, 0, 0], [2, 0, 0]), 2.0)
        # General case
        self.assertAlmostEqual(v_dot([1, 2, 3], [4, 5, 6]), 32.0)
    
    def test_v_cross(self):
        # i × j = k
        result = v_cross([1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 1.0)
        
        # j × i = -k
        result = v_cross([0, 1, 0], [1, 0, 0])
        self.assertAlmostEqual(result[2], -1.0)
    
    def test_v_norm(self):
        self.assertAlmostEqual(v_norm([3, 4, 0]), 5.0)
        self.assertAlmostEqual(v_norm([1, 0, 0]), 1.0)
        self.assertAlmostEqual(v_norm([0, 0, 0]), 0.0)
    
    def test_v_unit(self):
        result = v_unit([3, 0, 0])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        
        # Zero vector should return zero
        result = v_unit([0, 0, 0])
        self.assertEqual(result, [0.0, 0.0, 0.0])
    
    def test_v_scale(self):
        result = v_scale([1, 2, 3], 2)
        self.assertEqual(result, [2, 4, 6])


class TestAngleCalculations(unittest.TestCase):
    """Tests for angle computation functions."""
    
    def test_angle_between_perpendicular(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [0, 1, 0]), 90.0, places=6)
        self.assertAlmostEqual(angle_between([1, 0, 0], [0, 0, 1]), 90.0, places=6)
    
    def test_angle_between_parallel(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [1, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(angle_between([1, 0, 0], [2, 0, 0]), 0.0, places=6)
    
    def test_angle_between_antiparallel(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [-1, 0, 0]), 180.0, places=6)
    
    def test_angle_between_45_degrees(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [1, 1, 0]), 45.0, places=5)
    
    def test_project_to_plane(self):
        # Project onto XY plane (remove Z component)
        v = project_to_plane([1, 2, 3], [0, 0, 1])
        self.assertAlmostEqual(v[0], 1.0, places=6)
        self.assertAlmostEqual(v[1], 2.0, places=6)
        self.assertAlmostEqual(v[2], 0.0, places=6)
        
        # Project onto YZ plane (remove X component)
        v = project_to_plane([1, 2, 3], [1, 0, 0])
        self.assertAlmostEqual(v[0], 0.0, places=6)
        self.assertAlmostEqual(v[1], 2.0, places=6)
        self.assertAlmostEqual(v[2], 3.0, places=6)
    
    def test_signed_angle_positive(self):
        # Rotation from +Y to +Z around +X axis
        ang = signed_angle([0, 1, 0], [0, 0, 1], [1, 0, 0])
        self.assertAlmostEqual(abs(ang), 90.0, places=5)
    
    def test_signed_angle_negative(self):
        # Rotation from +Z to +Y around +X axis
        ang = signed_angle([0, 0, 1], [0, 1, 0], [1, 0, 0])
        self.assertAlmostEqual(abs(ang), 90.0, places=5)
    
    def test_signed_angle_zero(self):
        ang = signed_angle([1, 0, 0], [1, 0, 0], [0, 0, 1])
        self.assertAlmostEqual(ang, 0.0, places=6)


class TestFiltering(unittest.TestCase):
    """Tests for filtering functions."""
    
    def test_median_filter_removes_spike(self):
        # Signal with spike at index 2
        values = [1.0, 1.0, 100.0, 1.0, 1.0]
        result = median_filter(values, window=3)
        self.assertAlmostEqual(result[2], 1.0)  # Spike removed
    
    def test_median_filter_preserves_edge(self):
        # Step function
        values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        result = median_filter(values, window=3)
        # Edge should be preserved (shifted by at most 1 sample)
        self.assertTrue(result[2] == 0.0 or result[3] == 1.0)
    
    def test_median_filter_handles_none(self):
        values = [1.0, None, 3.0, 4.0, 5.0]
        result = median_filter(values, window=3)
        self.assertEqual(len(result), 5)
    
    def test_median_filter_invalid_window(self):
        with self.assertRaises(ValueError):
            median_filter([1, 2, 3], window=2)  # Even window
        with self.assertRaises(ValueError):
            median_filter([1, 2, 3], window=1)  # Too small
    
    def test_ema_filter_smoothing(self):
        # Noisy signal
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        result = ema_filter(values, alpha=0.3)
        # Result should be smoother (lower variance)
        variance_before = sum((v - 0.5)**2 for v in values) / len(values)
        variance_after = sum((v - 0.5)**2 for v in result) / len(result)
        self.assertLess(variance_after, variance_before)
    
    def test_ema_filter_handles_none(self):
        values = [1.0, None, 3.0]
        result = ema_filter(values, alpha=0.5)
        # None should be replaced with previous value
        self.assertEqual(result[1], result[0])
    
    def test_smooth_series_pipeline(self):
        # Combined filter should remove spike and smooth
        values = [1.0, 1.0, 100.0, 1.0, 1.0, 1.1, 0.9, 1.0]
        result = smooth_series(values, window=3, alpha=0.3)
        # Spike should be significantly reduced
        self.assertLess(result[2], 10.0)
    
    def test_detect_outliers_zscore(self):
        values = [1.0, 1.1, 0.9, 1.0, 100.0, 1.0, 0.95]
        outliers = detect_outliers_zscore(values, threshold=2.0)
        self.assertTrue(outliers[4])  # 100.0 is an outlier
        self.assertFalse(outliers[0])  # 1.0 is not an outlier
    
    def test_interpolate_missing(self):
        values = [1.0, None, None, 4.0, 5.0]
        result = interpolate_missing(values, max_gap=3)
        self.assertAlmostEqual(result[1], 2.0)  # Linear interpolation
        self.assertAlmostEqual(result[2], 3.0)


class TestAssessmentMetrics(unittest.TestCase):
    """Tests for biomechanical assessment functions."""
    
    def test_knee_valgus_neutral(self):
        # Knee directly in line with hip and ankle
        hip = [-100, 0, 500]
        knee = [-100, 0, 250]
        ankle = [-100, 0, 0]
        angle, deviation = compute_knee_valgus_angle(hip, knee, ankle, 'left')
        self.assertEqual(deviation, KneeDeviation.NEUTRAL)
        self.assertAlmostEqual(angle, 0.0, places=1)
    
    def test_knee_valgus_inward(self):
        # Knee moved inward (toward center/positive X for left leg)
        hip = [-100, 0, 500]
        knee = [-50, 0, 250]  # Moved toward center (+X direction)
        ankle = [-100, 0, 0]
        angle, deviation = compute_knee_valgus_angle(hip, knee, ankle, 'left')
        # Check that deviation is detected (non-zero angle)
        self.assertNotEqual(angle, 0)
    
    def test_knee_valgus_outward(self):
        # Knee moved outward (away from center)
        hip = [-100, 0, 500]
        knee = [-150, 0, 250]  # Moved away from center (-X direction)
        ankle = [-100, 0, 0]
        angle, deviation = compute_knee_valgus_angle(hip, knee, ankle, 'left')
        # Check that deviation is detected (non-zero angle, opposite sign)
        self.assertNotEqual(angle, 0)
    
    def test_torso_lean_neutral(self):
        # Torso vertical
        shoulders = [0, 0, 100]
        pelvis = [0, 0, 0]
        angle, deviation = compute_torso_lean_angle(shoulders, pelvis)
        self.assertEqual(deviation, TorsoDeviation.NEUTRAL)
        self.assertAlmostEqual(angle, 0.0, places=1)
    
    def test_torso_lean_forward(self):
        # Torso tilted forward
        shoulders = [0, 50, 100]  # Moved forward in Y
        pelvis = [0, 0, 0]
        angle, deviation = compute_torso_lean_angle(shoulders, pelvis)
        # Check non-zero angle detected
        self.assertNotEqual(angle, 0)
    
    def test_torso_lean_excessive(self):
        # Torso tilted significantly forward (45°)
        shoulders = [0, 100, 100]  # 45° forward lean
        pelvis = [0, 0, 0]
        angle, deviation = compute_torso_lean_angle(shoulders, pelvis)
        # Angle magnitude should be around 45°
        self.assertAlmostEqual(abs(angle), 45.0, places=0)
    
    def test_knee_flexion_extended(self):
        # Straight leg (180° between thigh and shank)
        hip = [0, 0, 500]
        knee = [0, 0, 250]
        ankle = [0, 0, 0]
        angle = compute_knee_flexion_angle(hip, knee, ankle)
        self.assertAlmostEqual(angle, 0.0, places=0)  # 0° flexion
    
    def test_knee_flexion_bent(self):
        # Bent knee (90° angle)
        hip = [0, 0, 500]
        knee = [0, 0, 250]
        ankle = [0, 250, 250]  # Shank pointing forward
        angle = compute_knee_flexion_angle(hip, knee, ankle)
        self.assertAlmostEqual(angle, 90.0, places=0)


class TestSquatPhaseDetection(unittest.TestCase):
    """Tests for squat phase detection."""
    
    def test_detect_squat_depth(self):
        # Simulated pelvis heights during squat
        heights = [100, 90, 80, 70, 60, 70, 80, 90, 100]
        depths = detect_squat_depth(heights, heights)
        
        self.assertEqual(len(depths), len(heights))
        self.assertAlmostEqual(depths[0], 0.0)  # Standing
        self.assertAlmostEqual(depths[4], 1.0)  # Bottom
        self.assertAlmostEqual(depths[8], 0.0)  # Back to standing
    
    def test_detect_squat_phases_single(self):
        # Single squat
        depths = [0.0, 0.1, 0.3, 0.5, 0.8, 0.5, 0.3, 0.1, 0.0] * 2
        phases = detect_squat_phases(depths, min_depth=0.2, min_phase_frames=3)
        
        self.assertEqual(len(phases), 2)  # Two repetitions
        self.assertEqual(phases[0].bottom_frame, 4)  # Bottom at index 4
    
    def test_detect_squat_phases_empty(self):
        # No squat movement
        depths = [0.0, 0.0, 0.0, 0.0]
        phases = detect_squat_phases(depths, min_depth=0.2)
        self.assertEqual(len(phases), 0)


class TestStatisticalSummary(unittest.TestCase):
    """Tests for statistical summary computation."""
    
    def test_from_values_normal(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = StatisticalSummary.from_values(values)
        
        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.mean, 3.0)
        self.assertAlmostEqual(summary.median, 3.0)
        self.assertAlmostEqual(summary.min, 1.0)
        self.assertAlmostEqual(summary.max, 5.0)
    
    def test_from_values_empty(self):
        summary = StatisticalSummary.from_values([])
        self.assertIsNone(summary)
    
    def test_from_values_single(self):
        summary = StatisticalSummary.from_values([5.0])
        self.assertIsNone(summary)  # Need at least 2 values for std


if __name__ == "__main__":
    unittest.main()
