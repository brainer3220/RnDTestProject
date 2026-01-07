"""
NASM Overhead Squat Assessment Module

This module implements biomechanically accurate assessment metrics for the
NASM (National Academy of Sports Medicine) Overhead Squat Assessment protocol.

References:
    - Clark, M. A., & Lucett, S. C. (2010). NASM Essentials of Corrective Exercise Training.
    - Hewett, T. E., et al. (2005). Biomechanical measures of neuromuscular control
      and valgus loading of the knee. American Journal of Sports Medicine.

Author: Research Team
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .geometry import (
    angle_between,
    project_to_plane,
    signed_angle,
    v_cross,
    v_dot,
    v_norm,
    v_scale,
    v_sub,
    v_unit,
)
from .filtering import smooth_series


# =============================================================================
# Type Definitions
# =============================================================================

Vector3 = List[float]  # [x, y, z]


class KneeDeviation(Enum):
    """Knee deviation direction in frontal plane."""
    VALGUS = "inward"   # Knee moves medially (toward midline)
    VARUS = "outward"   # Knee moves laterally (away from midline)
    NEUTRAL = "neutral"


class LumbarDeviation(Enum):
    """Lumbar spine deviation type."""
    EXCESSIVE_LORDOSIS = "excessive_arch"  # Hyperlordosis
    FLEXION = "flexion"                    # Loss of lordosis / kyphotic
    NEUTRAL = "neutral"


class TorsoDeviation(Enum):
    """Torso lean classification."""
    EXCESSIVE_FORWARD = "excessive_forward_lean"
    BACKWARD = "backward_lean"
    NEUTRAL = "neutral"


@dataclass
class FrameMetrics:
    """Single frame assessment metrics."""
    timestamp: float
    squat_depth: float  # Normalized depth (0=standing, 1=deepest)
    
    # Anterior View
    left_knee_valgus_angle: float   # Positive=valgus(inward), Negative=varus(outward)
    right_knee_valgus_angle: float
    left_knee_deviation: KneeDeviation
    right_knee_deviation: KneeDeviation
    
    # Lateral View
    torso_lean_angle: float         # Positive=forward, Negative=backward
    lumbar_angle: float             # Deviation from neutral (~170-180°)
    
    # Additional biomechanical metrics
    knee_flexion_left: float        # Sagittal plane knee angle
    knee_flexion_right: float
    hip_flexion_angle: float        # Hip hinge angle


@dataclass
class SquatPhase:
    """Squat phase boundaries detected from motion data."""
    start_frame: int
    bottom_frame: int   # Maximum depth point
    end_frame: int
    max_depth: float


@dataclass
class StatisticalSummary:
    """Statistical summary for a metric."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    p5: float   # 5th percentile
    p95: float  # 95th percentile
    n: int
    
    @classmethod
    def from_values(cls, values: List[float]) -> Optional['StatisticalSummary']:
        if not values or len(values) < 2:
            return None
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return cls(
            mean=statistics.mean(values),
            std=statistics.stdev(values),
            min=min(values),
            max=max(values),
            median=statistics.median(values),
            p5=sorted_vals[max(0, int(0.05 * (n - 1)))],
            p95=sorted_vals[min(n - 1, int(0.95 * (n - 1)))],
            n=n
        )


@dataclass
class NASMAssessmentResult:
    """Complete NASM overhead squat assessment result."""
    subject_id: str
    total_frames: int
    valid_frames: int
    squat_phases: List[SquatPhase]
    
    # Frame-by-frame metrics
    frame_metrics: List[FrameMetrics]
    
    # Aggregated statistics (computed at bottom of squat)
    left_knee_valgus: StatisticalSummary
    right_knee_valgus: StatisticalSummary
    torso_lean: StatisticalSummary
    lumbar_deviation: StatisticalSummary
    
    # Classification results
    left_knee_classification: KneeDeviation
    right_knee_classification: KneeDeviation
    torso_classification: TorsoDeviation
    lumbar_classification: LumbarDeviation
    
    # Clinical thresholds exceeded
    compensation_flags: List[str] = field(default_factory=list)


# =============================================================================
# Constants & Thresholds (Based on NASM Guidelines)
# =============================================================================

# Knee valgus/varus thresholds (degrees)
KNEE_VALGUS_THRESHOLD = 10.0    # >10° inward = compensation
KNEE_VARUS_THRESHOLD = -10.0    # <-10° outward = compensation

# Torso lean thresholds (degrees from vertical)
TORSO_FORWARD_LEAN_THRESHOLD = 30.0   # Excessive forward lean
TORSO_BACKWARD_LEAN_THRESHOLD = -10.0  # Backward lean

# Lumbar angle thresholds (degrees)
LUMBAR_NEUTRAL_MIN = 160.0     # Normal lordosis range
LUMBAR_NEUTRAL_MAX = 180.0
LUMBAR_HYPERLORDOSIS = 150.0   # Excessive arch

# Squat depth threshold (normalized)
SQUAT_DEPTH_THRESHOLD = 0.3    # Consider "bottom" of squat when depth > 30%


# =============================================================================
# Utility Functions
# =============================================================================

def _point(row_map: Dict[str, Optional[float]], name: str) -> Optional[Vector3]:
    """Extract 3D point from row data with validation."""
    x = row_map.get(f"{name}_x")
    y = row_map.get(f"{name}_y")
    z = row_map.get(f"{name}_z")
    if x is None or y is None or z is None:
        return None
    if any(math.isnan(v) if isinstance(v, float) else False for v in [x, y, z]):
        return None
    return [float(x), float(y), float(z)]


def _midpoint(a: Vector3, b: Vector3) -> Vector3:
    """Calculate midpoint between two 3D points."""
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]


def _compute_body_frame(
    l_hip: Vector3, 
    r_hip: Vector3, 
    l_shoulder: Vector3, 
    r_shoulder: Vector3
) -> Tuple[Vector3, Vector3, Vector3, Vector3]:
    """
    Compute anatomically-aligned body coordinate frame.
    
    Returns:
        Tuple of (origin, x_axis, y_axis, z_axis) where:
        - origin: pelvis center (midpoint of hips)
        - x_axis: pointing RIGHT (lateral)
        - y_axis: pointing FORWARD (anterior)  
        - z_axis: pointing UP (superior)
    
    This follows ISB (International Society of Biomechanics) conventions.
    """
    pelvis = _midpoint(l_hip, r_hip)
    shoulders = _midpoint(l_shoulder, r_shoulder)
    
    # X-axis: right direction (from left hip to right hip)
    x_axis = v_unit(v_sub(r_hip, l_hip))
    
    # Preliminary Z-axis: up direction (from pelvis to shoulders)
    z_preliminary = v_unit(v_sub(shoulders, pelvis))
    
    # Y-axis: forward direction (cross product of z × x)
    y_axis = v_unit(v_cross(z_preliminary, x_axis))
    
    # Re-orthogonalize Z-axis (cross product of x × y)
    z_axis = v_unit(v_cross(x_axis, y_axis))
    
    # Final re-orthogonalization of X-axis
    x_axis = v_unit(v_cross(y_axis, z_axis))
    
    return pelvis, x_axis, y_axis, z_axis


def _transform_to_local(
    point: Vector3,
    origin: Vector3,
    x_axis: Vector3,
    y_axis: Vector3,
    z_axis: Vector3
) -> Vector3:
    """Transform a global point to local body coordinate frame."""
    v = v_sub(point, origin)
    return [v_dot(v, x_axis), v_dot(v, y_axis), v_dot(v, z_axis)]


# =============================================================================
# Biomechanical Angle Computations
# =============================================================================

def compute_knee_valgus_angle(
    hip: Vector3, 
    knee: Vector3, 
    ankle: Vector3,
    side: str  # 'left' or 'right'
) -> Tuple[float, KneeDeviation]:
    """
    Compute knee valgus/varus angle in the frontal (coronal) plane.
    
    The frontal plane view (Anterior View) shows medial/lateral knee deviation.
    We project the thigh and shank vectors onto the X-Z plane (frontal plane)
    and measure the angle between the knee position and the hip-ankle line.
    
    Positive angle = Valgus (knee moves inward/medially)
    Negative angle = Varus (knee moves outward/laterally)
    
    Args:
        hip: Hip joint position in local coordinates [x, y, z]
        knee: Knee joint position in local coordinates
        ankle: Ankle joint position in local coordinates
        side: 'left' or 'right' leg
    
    Returns:
        Tuple of (angle in degrees, deviation classification)
    """
    # Project to frontal plane (X-Z plane, remove Y component)
    hip_frontal = [hip[0], 0.0, hip[2]]
    knee_frontal = [knee[0], 0.0, knee[2]]
    ankle_frontal = [ankle[0], 0.0, ankle[2]]
    
    # Calculate the midpoint of hip-ankle line in frontal plane
    # This represents the "ideal" neutral knee position
    ideal_knee_x = (hip_frontal[0] + ankle_frontal[0]) / 2
    
    # Knee deviation from ideal line (medial/lateral displacement)
    knee_offset = knee_frontal[0] - ideal_knee_x
    
    # Calculate thigh and shank vectors
    v_thigh = v_sub(hip_frontal, knee_frontal)
    v_shank = v_sub(ankle_frontal, knee_frontal)
    
    # Get the angle between thigh and shank in frontal plane
    # Using signed angle with Y-axis as normal (pointing forward)
    normal = [0.0, 1.0, 0.0]  # Normal to frontal plane
    
    # Calculate signed angle
    angle = signed_angle(v_thigh, v_shank, normal)
    
    # Determine direction based on side and knee offset
    # For LEFT leg: positive offset = lateral (varus), negative = medial (valgus)
    # For RIGHT leg: positive offset = medial (valgus), negative = lateral (varus)
    if side == 'left':
        # Left leg: negative X is medial (toward center)
        valgus_angle = -knee_offset  # Invert so positive = valgus
    else:
        # Right leg: positive X is medial (toward center)
        valgus_angle = knee_offset
    
    # Scale by the thigh-shank angle magnitude for clinical relevance
    # Normalize by typical leg length ratio
    thigh_length = v_norm(v_sub(hip, knee))
    if thigh_length > 0:
        valgus_angle = math.degrees(math.atan2(valgus_angle, thigh_length))
    else:
        valgus_angle = 0.0
    
    # Classify deviation
    if valgus_angle > KNEE_VALGUS_THRESHOLD:
        deviation = KneeDeviation.VALGUS
    elif valgus_angle < KNEE_VARUS_THRESHOLD:
        deviation = KneeDeviation.VARUS
    else:
        deviation = KneeDeviation.NEUTRAL
    
    return valgus_angle, deviation


def compute_knee_flexion_angle(
    hip: Vector3,
    knee: Vector3,
    ankle: Vector3
) -> float:
    """
    Compute knee flexion angle in the sagittal plane.
    
    0° = fully extended knee
    Higher values = more flexion (bent knee)
    """
    v_thigh = v_sub(hip, knee)
    v_shank = v_sub(ankle, knee)
    
    # Project to sagittal plane (Y-Z plane)
    v_thigh_sag = [0.0, v_thigh[1], v_thigh[2]]
    v_shank_sag = [0.0, v_shank[1], v_shank[2]]
    
    angle = angle_between(v_thigh_sag, v_shank_sag)
    return 180.0 - angle  # Convert to flexion angle


def compute_torso_lean_angle_global(
    shoulders: Vector3,
    pelvis: Vector3,
    ankles: Vector3
) -> Tuple[float, TorsoDeviation]:
    """
    Compute torso forward/backward lean angle using global coordinates.
    
    Measures the angle of the pelvis-shoulders line relative to vertical,
    viewed from the side (lateral view). Uses ankle position as ground reference.
    
    This approach works regardless of body coordinate frame definition.
    
    Note: This function auto-detects the vertical axis from the data.
    In typical motion capture setups:
    - If shoulder_Y > pelvis_Y significantly: Y is vertical
    - If shoulder_Z > pelvis_Z significantly: Z is vertical
    
    Positive angle = Forward lean (shoulders ahead of pelvis)
    Negative angle = Backward lean
    
    Args:
        shoulders: Shoulder center position in GLOBAL coordinates
        pelvis: Pelvis center position in GLOBAL coordinates
        ankles: Ankle midpoint in GLOBAL coordinates (ground reference)
    
    Returns:
        Tuple of (angle in degrees, deviation classification)
    """
    # Torso vector from pelvis to shoulders (in global coords)
    torso_vec = v_sub(shoulders, pelvis)
    
    # Auto-detect vertical axis:
    # The axis with the largest difference between shoulders and pelvis is likely vertical
    dy = abs(torso_vec[1])  # Y difference
    dz = abs(torso_vec[2])  # Z difference
    
    if dy > dz:
        # Y is vertical (common in some motion capture systems)
        # Sagittal plane = X-Y plane
        # Project to X-Y plane (remove Z component)
        torso_sagittal = [torso_vec[0], torso_vec[1], 0.0]
        vertical = [0.0, 1.0, 0.0]  # Y is up
        normal = [0.0, 0.0, 1.0]    # Z is normal (depth direction)
    else:
        # Z is vertical (standard biomechanics convention)
        # Sagittal plane = Y-Z plane
        torso_sagittal = [0.0, torso_vec[1], torso_vec[2]]
        vertical = [0.0, 0.0, 1.0]  # Z is up
        normal = [1.0, 0.0, 0.0]    # X is normal
    
    # Signed angle: positive when torso tilts forward
    angle = signed_angle(vertical, torso_sagittal, normal)
    
    # Classify deviation
    if angle > TORSO_FORWARD_LEAN_THRESHOLD:
        deviation = TorsoDeviation.EXCESSIVE_FORWARD
    elif angle < TORSO_BACKWARD_LEAN_THRESHOLD:
        deviation = TorsoDeviation.BACKWARD
    else:
        deviation = TorsoDeviation.NEUTRAL
    
    return angle, deviation


def compute_torso_lean_angle(
    shoulders: Vector3,
    pelvis: Vector3
) -> Tuple[float, TorsoDeviation]:
    """
    Compute torso forward/backward lean angle in the sagittal plane.
    
    NOTE: This function is for local coordinates. For global coordinates,
    use compute_torso_lean_angle_global() instead.
    
    The torso vector (pelvis to shoulders) is projected onto the sagittal
    plane (Y-Z plane) and compared to the vertical axis.
    
    Positive angle = Forward lean
    Negative angle = Backward lean
    
    Args:
        shoulders: Shoulder center position in local coordinates
        pelvis: Pelvis center position in local coordinates
    
    Returns:
        Tuple of (angle in degrees, deviation classification)
    """
    # Torso vector from pelvis to shoulders
    torso_vec = v_sub(shoulders, pelvis)
    
    # Project to sagittal plane (Y-Z plane, remove X component)
    torso_sagittal = [0.0, torso_vec[1], torso_vec[2]]
    
    # Vertical reference vector (pointing up)
    vertical = [0.0, 0.0, 1.0]
    
    # Normal vector for sagittal plane (pointing right/lateral)
    normal = [1.0, 0.0, 0.0]
    
    # Signed angle: positive when torso tilts forward (toward +Y)
    angle = signed_angle(vertical, torso_sagittal, normal)
    
    # Classify deviation
    if angle > TORSO_FORWARD_LEAN_THRESHOLD:
        deviation = TorsoDeviation.EXCESSIVE_FORWARD
    elif angle < TORSO_BACKWARD_LEAN_THRESHOLD:
        deviation = TorsoDeviation.BACKWARD
    else:
        deviation = TorsoDeviation.NEUTRAL
    
    return angle, deviation


def compute_lumbar_angle(
    upper_spine: Vector3,  # torso/thoracic
    pelvis: Vector3,
    waist: Vector3  # L3-L5 region approximation
) -> Tuple[float, LumbarDeviation]:
    """
    Compute lumbar spine angle (lordosis/kyphosis assessment).
    
    Measures the angle at the waist/lumbar region between:
    - Upper vector: waist to thoracic spine
    - Lower vector: waist to pelvis
    
    This approximates lumbar curvature in the sagittal plane.
    
    Normal lordosis: 160-180° (slightly curved)
    Hyperlordosis (excessive arch): <150°
    Flexion (flattened/kyphotic): >180° or reversed curve
    
    Args:
        upper_spine: Upper spine/torso marker position
        pelvis: Pelvis center position
        waist: Lumbar region marker position
    
    Returns:
        Tuple of (angle in degrees, deviation classification)
    """
    # Vectors from waist
    v_upper = v_sub(upper_spine, waist)
    v_lower = v_sub(pelvis, waist)
    
    # Project to sagittal plane (Y-Z)
    v_upper_sag = [0.0, v_upper[1], v_upper[2]]
    v_lower_sag = [0.0, v_lower[1], v_lower[2]]
    
    # Normal for sagittal plane
    normal = [1.0, 0.0, 0.0]
    
    # Compute angle (using signed angle for direction)
    angle = signed_angle(v_lower_sag, v_upper_sag, normal)
    
    # Convert to positive angle representation
    if angle < 0:
        angle = 360.0 + angle
    
    # Map to clinical range (typically 150-180 for lordosis assessment)
    # Normalize angle to be in reasonable range
    if angle > 180:
        angle = 360 - angle
    
    # Classify deviation
    if angle < LUMBAR_HYPERLORDOSIS:
        deviation = LumbarDeviation.EXCESSIVE_LORDOSIS
    elif angle < LUMBAR_NEUTRAL_MIN:
        deviation = LumbarDeviation.EXCESSIVE_LORDOSIS
    elif angle > LUMBAR_NEUTRAL_MAX:
        deviation = LumbarDeviation.FLEXION
    else:
        deviation = LumbarDeviation.NEUTRAL
    
    return angle, deviation


def compute_hip_flexion_angle(
    shoulders: Vector3,
    pelvis: Vector3,
    knee: Vector3  # Average of both knees
) -> float:
    """
    Compute hip flexion angle for squat depth assessment.
    
    Measures the angle at the hip between torso and thigh.
    """
    v_torso = v_sub(shoulders, pelvis)
    v_thigh = v_sub(knee, pelvis)
    
    return angle_between(v_torso, v_thigh)


# =============================================================================
# Squat Phase Detection
# =============================================================================

def detect_squat_depth(
    pelvis_heights: List[float],
    knee_heights: List[float]
) -> List[float]:
    """
    Compute normalized squat depth for each frame.
    
    Depth is normalized where:
    - 0.0 = standing (maximum height)
    - 1.0 = deepest squat position
    
    Uses pelvis height relative to standing position.
    """
    if not pelvis_heights:
        return []
    
    max_height = max(pelvis_heights)
    min_height = min(pelvis_heights)
    height_range = max_height - min_height
    
    if height_range < 1e-6:
        return [0.0] * len(pelvis_heights)
    
    # Normalized depth (inverted so lower = higher depth value)
    return [(max_height - h) / height_range for h in pelvis_heights]


def detect_squat_phases(
    depths: List[float],
    min_depth: float = 0.2,
    min_phase_frames: int = 10
) -> List[SquatPhase]:
    """
    Detect individual squat repetitions from depth signal.
    
    A squat phase is defined as:
    1. Start: depth crosses above threshold (descending)
    2. Bottom: maximum depth point
    3. End: depth crosses below threshold (ascending)
    """
    phases = []
    n = len(depths)
    i = 0
    
    while i < n:
        # Find start of squat (entering threshold)
        while i < n and depths[i] < min_depth:
            i += 1
        if i >= n:
            break
        
        start = i
        max_depth = depths[i]
        bottom = i
        
        # Find bottom and end of squat
        while i < n and depths[i] >= min_depth:
            if depths[i] > max_depth:
                max_depth = depths[i]
                bottom = i
            i += 1
        
        end = i - 1
        
        # Validate phase
        if end - start >= min_phase_frames:
            phases.append(SquatPhase(
                start_frame=start,
                bottom_frame=bottom,
                end_frame=end,
                max_depth=max_depth
            ))
    
    return phases


# =============================================================================
# Data Processing
# =============================================================================

def smooth_rows(
    headers: List[str], 
    rows: List[List[Optional[float]]], 
    window: int = 5, 
    alpha: float = 0.2
) -> List[List[Optional[float]]]:
    """
    Apply noise filtering to all coordinate columns.
    
    Uses a two-stage filter:
    1. Median filter: removes spike noise while preserving edges
    2. Exponential moving average: smooths remaining high-frequency noise
    """
    if not rows:
        return []
    
    columns = list(zip(*rows))
    smoothed_cols = []
    
    for col in columns:
        col_vals = [
            v if v is not None and not (isinstance(v, float) and math.isnan(v)) 
            else None 
            for v in col
        ]
        smoothed_cols.append(smooth_series(col_vals, window=window, alpha=alpha))
    
    return [list(row) for row in zip(*smoothed_cols)]


def maybe_filter_outliers(
    headers: List[str],
    rows: List[List[Optional[float]]],
    joints: List[str],
    contamination: float = 0.02
) -> List[List[Optional[float]]]:
    """
    Remove outlier frames using velocity-based anomaly detection.
    
    Uses IsolationForest on joint velocities to detect physically
    impossible movements (sensor glitches).
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return rows
    
    if len(rows) < 10:
        return rows
    
    header_index = {h: i for i, h in enumerate(headers)}
    
    # Compute velocities (frame-to-frame differences)
    velocities = []
    for i in range(1, len(rows)):
        frame_vel = []
        for joint in joints:
            for axis in ('x', 'y', 'z'):
                key = f"{joint}_{axis}"
                idx = header_index.get(key)
                if idx is None:
                    frame_vel.append(0.0)
                    continue
                
                curr = rows[i][idx]
                prev = rows[i-1][idx]
                
                if curr is None or prev is None:
                    frame_vel.append(0.0)
                elif isinstance(curr, float) and math.isnan(curr):
                    frame_vel.append(0.0)
                elif isinstance(prev, float) and math.isnan(prev):
                    frame_vel.append(0.0)
                else:
                    frame_vel.append(float(curr) - float(prev))
        
        velocities.append(frame_vel)
    
    if not velocities:
        return rows
    
    # Fit anomaly detector
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = model.fit_predict(velocities)
    
    # Keep first frame and frames with normal velocity
    filtered = [rows[0]]
    for i, pred in enumerate(preds):
        if pred == 1:  # Normal
            filtered.append(rows[i + 1])
    
    return filtered


# =============================================================================
# Main Assessment Function
# =============================================================================

def assess_frames(
    headers: List[str],
    rows: List[List[Optional[float]]],
    subject_id: str = "unknown"
) -> NASMAssessmentResult:
    """
    Perform complete NASM overhead squat assessment on motion capture data.
    
    Args:
        headers: Column headers from motion capture data
        rows: Frame data (each row is one frame, values are joint coordinates)
        subject_id: Identifier for the subject
    
    Returns:
        NASMAssessmentResult with complete assessment metrics and classifications
    """
    header_index = {h: i for i, h in enumerate(headers)}
    timestamp_idx = header_index.get('timestamp')
    
    def row_to_map(row):
        return {h: row[header_index[h]] if h in header_index else None for h in header_index}
    
    frame_metrics: List[FrameMetrics] = []
    pelvis_heights: List[float] = []
    knee_heights: List[float] = []
    valid_frames = 0
    
    # Process each frame
    for frame_idx, row in enumerate(rows):
        row_map = row_to_map(row)
        
        # Extract all required joints
        l_hip = _point(row_map, "l_hip")
        r_hip = _point(row_map, "r_hip")
        l_shoulder = _point(row_map, "l_shoulder")
        r_shoulder = _point(row_map, "r_shoulder")
        l_knee = _point(row_map, "l_knee")
        r_knee = _point(row_map, "r_knee")
        l_ankle = _point(row_map, "l_ankle")
        r_ankle = _point(row_map, "r_ankle")
        torso = _point(row_map, "torso")
        waist = _point(row_map, "waist")
        
        # Skip frames with missing data
        required = [l_hip, r_hip, l_shoulder, r_shoulder, l_knee, r_knee, 
                   l_ankle, r_ankle, torso, waist]
        if any(p is None for p in required):
            continue
        
        valid_frames += 1
        
        # Compute body coordinate frame
        pelvis, x_axis, y_axis, z_axis = _compute_body_frame(
            l_hip, r_hip, l_shoulder, r_shoulder
        )
        
        # Transform all points to local coordinates
        l_hip_l = _transform_to_local(l_hip, pelvis, x_axis, y_axis, z_axis)
        r_hip_l = _transform_to_local(r_hip, pelvis, x_axis, y_axis, z_axis)
        l_knee_l = _transform_to_local(l_knee, pelvis, x_axis, y_axis, z_axis)
        r_knee_l = _transform_to_local(r_knee, pelvis, x_axis, y_axis, z_axis)
        l_ankle_l = _transform_to_local(l_ankle, pelvis, x_axis, y_axis, z_axis)
        r_ankle_l = _transform_to_local(r_ankle, pelvis, x_axis, y_axis, z_axis)
        shoulders_l = _transform_to_local(
            _midpoint(l_shoulder, r_shoulder), pelvis, x_axis, y_axis, z_axis
        )
        torso_l = _transform_to_local(torso, pelvis, x_axis, y_axis, z_axis)
        waist_l = _transform_to_local(waist, pelvis, x_axis, y_axis, z_axis)
        
        # Record heights for squat phase detection
        pelvis_heights.append(pelvis[2])  # Global Z coordinate
        knee_heights.append((l_knee[2] + r_knee[2]) / 2)
        
        # Compute assessment metrics
        # 1. Knee valgus/varus (Anterior View)
        left_valgus, left_dev = compute_knee_valgus_angle(
            l_hip_l, l_knee_l, l_ankle_l, 'left'
        )
        right_valgus, right_dev = compute_knee_valgus_angle(
            r_hip_l, r_knee_l, r_ankle_l, 'right'
        )
        
        # 2. Knee flexion angles
        left_flexion = compute_knee_flexion_angle(l_hip_l, l_knee_l, l_ankle_l)
        right_flexion = compute_knee_flexion_angle(r_hip_l, r_knee_l, r_ankle_l)
        
        # 3. Torso lean (Lateral View) - Use GLOBAL coordinates
        # Global coordinate system: Y = anterior-posterior, Z = vertical
        shoulders_global = _midpoint(l_shoulder, r_shoulder)
        ankles_global = _midpoint(l_ankle, r_ankle)
        torso_lean, torso_dev = compute_torso_lean_angle_global(
            shoulders_global, pelvis, ankles_global
        )
        
        # 4. Lumbar angle (Lateral View)
        lumbar, lumbar_dev = compute_lumbar_angle(torso_l, [0.0, 0.0, 0.0], waist_l)
        
        # 5. Hip flexion
        knee_avg = _midpoint(l_knee_l, r_knee_l)
        hip_flex = compute_hip_flexion_angle(shoulders_l, [0.0, 0.0, 0.0], knee_avg)
        
        # Get timestamp
        timestamp = row[timestamp_idx] if timestamp_idx is not None else float(frame_idx)
        
        frame_metrics.append(FrameMetrics(
            timestamp=float(timestamp) if timestamp else float(frame_idx),
            squat_depth=0.0,  # Will be computed after all frames
            left_knee_valgus_angle=left_valgus,
            right_knee_valgus_angle=right_valgus,
            left_knee_deviation=left_dev,
            right_knee_deviation=right_dev,
            torso_lean_angle=torso_lean,
            lumbar_angle=lumbar,
            knee_flexion_left=left_flexion,
            knee_flexion_right=right_flexion,
            hip_flexion_angle=hip_flex
        ))
    
    # Compute squat depths
    depths = detect_squat_depth(pelvis_heights, knee_heights)
    for i, depth in enumerate(depths):
        if i < len(frame_metrics):
            frame_metrics[i] = FrameMetrics(
                timestamp=frame_metrics[i].timestamp,
                squat_depth=depth,
                left_knee_valgus_angle=frame_metrics[i].left_knee_valgus_angle,
                right_knee_valgus_angle=frame_metrics[i].right_knee_valgus_angle,
                left_knee_deviation=frame_metrics[i].left_knee_deviation,
                right_knee_deviation=frame_metrics[i].right_knee_deviation,
                torso_lean_angle=frame_metrics[i].torso_lean_angle,
                lumbar_angle=frame_metrics[i].lumbar_angle,
                knee_flexion_left=frame_metrics[i].knee_flexion_left,
                knee_flexion_right=frame_metrics[i].knee_flexion_right,
                hip_flexion_angle=frame_metrics[i].hip_flexion_angle
            )
    
    # Detect squat phases
    squat_phases = detect_squat_phases(depths)
    
    # Extract metrics at bottom of squat (most clinically relevant)
    bottom_indices = set()
    for phase in squat_phases:
        # Include frames near bottom (±5 frames)
        for j in range(max(0, phase.bottom_frame - 5), 
                       min(len(frame_metrics), phase.bottom_frame + 6)):
            bottom_indices.add(j)
    
    # If no clear squat phases, use frames with depth > threshold
    if not bottom_indices:
        bottom_indices = {i for i, fm in enumerate(frame_metrics) 
                         if fm.squat_depth > SQUAT_DEPTH_THRESHOLD}
    
    # If still no frames, use all frames
    if not bottom_indices:
        bottom_indices = set(range(len(frame_metrics)))
    
    # Compute statistics from bottom-of-squat frames
    bottom_frames = [frame_metrics[i] for i in sorted(bottom_indices) 
                     if i < len(frame_metrics)]
    
    left_valgus_vals = [f.left_knee_valgus_angle for f in bottom_frames]
    right_valgus_vals = [f.right_knee_valgus_angle for f in bottom_frames]
    torso_vals = [f.torso_lean_angle for f in bottom_frames]
    lumbar_vals = [f.lumbar_angle for f in bottom_frames]
    
    # Compute statistical summaries
    left_valgus_summary = StatisticalSummary.from_values(left_valgus_vals)
    right_valgus_summary = StatisticalSummary.from_values(right_valgus_vals)
    torso_summary = StatisticalSummary.from_values(torso_vals)
    lumbar_summary = StatisticalSummary.from_values(lumbar_vals)
    
    # Classification based on mean values
    def classify_knee(summary: Optional[StatisticalSummary]) -> KneeDeviation:
        if summary is None:
            return KneeDeviation.NEUTRAL
        if summary.mean > KNEE_VALGUS_THRESHOLD:
            return KneeDeviation.VALGUS
        elif summary.mean < KNEE_VARUS_THRESHOLD:
            return KneeDeviation.VARUS
        return KneeDeviation.NEUTRAL
    
    def classify_torso(summary: Optional[StatisticalSummary]) -> TorsoDeviation:
        if summary is None:
            return TorsoDeviation.NEUTRAL
        if summary.mean > TORSO_FORWARD_LEAN_THRESHOLD:
            return TorsoDeviation.EXCESSIVE_FORWARD
        elif summary.mean < TORSO_BACKWARD_LEAN_THRESHOLD:
            return TorsoDeviation.BACKWARD
        return TorsoDeviation.NEUTRAL
    
    def classify_lumbar(summary: Optional[StatisticalSummary]) -> LumbarDeviation:
        if summary is None:
            return LumbarDeviation.NEUTRAL
        if summary.mean < LUMBAR_HYPERLORDOSIS:
            return LumbarDeviation.EXCESSIVE_LORDOSIS
        elif summary.mean > LUMBAR_NEUTRAL_MAX:
            return LumbarDeviation.FLEXION
        return LumbarDeviation.NEUTRAL
    
    # Identify compensation patterns
    compensation_flags = []
    
    if left_valgus_summary and left_valgus_summary.p95 > KNEE_VALGUS_THRESHOLD:
        compensation_flags.append("Left knee valgus (knees move inward)")
    if right_valgus_summary and right_valgus_summary.p95 > KNEE_VALGUS_THRESHOLD:
        compensation_flags.append("Right knee valgus (knees move inward)")
    if left_valgus_summary and left_valgus_summary.p5 < KNEE_VARUS_THRESHOLD:
        compensation_flags.append("Left knee varus (knees move outward)")
    if right_valgus_summary and right_valgus_summary.p5 < KNEE_VARUS_THRESHOLD:
        compensation_flags.append("Right knee varus (knees move outward)")
    if torso_summary and torso_summary.p95 > TORSO_FORWARD_LEAN_THRESHOLD:
        compensation_flags.append("Excessive forward lean")
    if lumbar_summary and lumbar_summary.p5 < LUMBAR_HYPERLORDOSIS:
        compensation_flags.append("Low back arches (excessive lordosis)")
    
    # Create default summaries if None
    default_summary = StatisticalSummary(0, 0, 0, 0, 0, 0, 0, 0)
    
    return NASMAssessmentResult(
        subject_id=subject_id,
        total_frames=len(rows),
        valid_frames=valid_frames,
        squat_phases=squat_phases,
        frame_metrics=frame_metrics,
        left_knee_valgus=left_valgus_summary or default_summary,
        right_knee_valgus=right_valgus_summary or default_summary,
        torso_lean=torso_summary or default_summary,
        lumbar_deviation=lumbar_summary or default_summary,
        left_knee_classification=classify_knee(left_valgus_summary),
        right_knee_classification=classify_knee(right_valgus_summary),
        torso_classification=classify_torso(torso_summary),
        lumbar_classification=classify_lumbar(lumbar_summary),
        compensation_flags=compensation_flags
    )


# =============================================================================
# Legacy API (Backward Compatibility)
# =============================================================================

def summarize_metric(values: List[float]) -> Optional[Dict[str, float]]:
    """Legacy function for backward compatibility."""
    summary = StatisticalSummary.from_values(values)
    if summary is None:
        return None
    return {
        "mean": summary.mean,
        "max": summary.max,
        "min": summary.min,
        "p95": summary.p95,
        "std": summary.std,
    }
