# NASM Overhead Squat Assessment Algorithm

## Overview

This module implements a comprehensive biomechanical analysis system for the NASM (National Academy of Sports Medicine) Overhead Squat Assessment protocol. The system processes noisy 3D motion capture data to extract clinically meaningful movement quality metrics.

## References

- Clark, M. A., & Lucett, S. C. (2010). *NASM Essentials of Corrective Exercise Training*
- Hewett, T. E., et al. (2005). Biomechanical measures of neuromuscular control and valgus loading of the knee. *American Journal of Sports Medicine*
- Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement*
- Robertson, D.G.E. et al. (2013). *Research Methods in Biomechanics*

---

## Data Model

### Input Format
Each frame contains 3D coordinates (x, y, z in mm) for anatomical landmarks:
- **Head/Trunk**: `head`, `torso`, `waist`
- **Upper extremity**: `l/r_shoulder`, `l/r_elbow`, `l/r_wrist`
- **Lower extremity**: `l/r_hip`, `l/r_knee`, `l/r_ankle`

### Coordinate System
The raw data uses a global coordinate system where:
- **X-axis**: Lateral (left-right)
- **Y-axis**: Anterior-posterior (front-back)
- **Z-axis**: Vertical (up-down), values ~2000mm indicate standing height

---

## Signal Processing Pipeline

### Stage 1: Outlier Detection (Optional)
Uses IsolationForest on frame-to-frame velocity magnitudes to detect sensor glitches:
- Contamination rate: 2%
- Features: Joint velocities for key landmarks
- Effect: Removes physically impossible frame transitions

### Stage 2: Median Filter
Removes impulse noise (spikes) while preserving motion edges:
```
window = 5 (frames)
For each sample:
    1. Collect values in ±2 frame window
    2. Sort and take median
    3. Replace original value
```
**Rationale**: At 30-60 Hz capture rates, a 5-frame window (~100-170ms) is sufficient to remove sensor glitches without distorting rapid movements.

### Stage 3: Exponential Moving Average (EMA)
Smooths high-frequency jitter:
```
α = 0.2 (smoothing factor)
y[n] = α·x[n] + (1-α)·y[n-1]
```
**Rationale**: α=0.2 provides gentle smoothing appropriate for human movement dynamics.

### Alternative: Butterworth Low-Pass Filter
For more precise biomechanical analysis:
- Cutoff frequency: 6 Hz (standard for human movement)
- Order: 2 (second-order)
- Implementation: Zero-phase (forward-backward) filtering

---

## Coordinate Transformation

### Body-Centric Reference Frame
To eliminate global orientation dependencies, all measurements are computed in a body-centric coordinate system following ISB (International Society of Biomechanics) conventions:

```
Origin: Pelvis center = (L_hip + R_hip) / 2

X-axis (Lateral):     L_hip → R_hip (pointing right)
Y-axis (Anterior):    Cross(Z, X) (pointing forward)
Z-axis (Superior):    Pelvis → Shoulder_center (pointing up)
```

### Orthonormalization
The axes are iteratively orthonormalized using Gram-Schmidt:
1. Compute preliminary Z from pelvis-to-shoulders vector
2. Compute Y = Z × X
3. Re-orthogonalize Z = X × Y
4. Final X = Y × Z

---

## Assessment Metrics

### 1. Knee Valgus/Varus Angle (Anterior View)

**Clinical Significance**: Excessive knee valgus during squatting is associated with:
- ACL injury risk (Hewett et al., 2005)
- Patellofemoral pain syndrome
- Hip abductor weakness

**Computation Method**:
1. Project hip, knee, ankle to frontal plane (X-Z plane, remove Y)
2. Calculate ideal knee position: midpoint of hip-ankle line
3. Measure knee medial/lateral offset from ideal
4. Convert to angular measure using thigh length normalization

```python
ideal_knee_x = (hip_x + ankle_x) / 2
offset = knee_x - ideal_knee_x

# For left leg: negative offset = medial (valgus)
# For right leg: positive offset = medial (valgus)
valgus_angle = arctan(offset / thigh_length)
```

**Thresholds**:
- Neutral: -10° to +10°
- Valgus (inward): > +10°
- Varus (outward): < -10°

### 2. Torso Lean Angle (Lateral View)

**Clinical Significance**: Excessive forward lean indicates:
- Hip flexor tightness
- Ankle dorsiflexion limitation
- Core/erector spinae weakness

**Computation Method**:
1. Compute torso vector: Pelvis → Shoulder_center
2. Project to sagittal plane (Y-Z plane, remove X)
3. Calculate signed angle from vertical

```python
torso_vec = shoulder_center - pelvis
torso_sagittal = [0, torso_vec.y, torso_vec.z]
vertical = [0, 0, 1]
lean_angle = signed_angle(vertical, torso_sagittal, [1, 0, 0])
```

**Thresholds**:
- Neutral: -10° to +30°
- Excessive forward: > +30°
- Backward lean: < -10°

### 3. Lumbar Arch Angle (Lateral View)

**Clinical Significance**: Excessive lumbar lordosis (arching) indicates:
- Hip flexor tightness
- Weak abdominals/gluteals
- Anterior pelvic tilt

**Computation Method**:
1. Compute upper spine vector: Waist → Torso
2. Compute lower spine vector: Waist → Pelvis
3. Project both to sagittal plane
4. Calculate angle between vectors

```python
v_upper = torso - waist
v_lower = pelvis - waist
# Project to Y-Z plane
angle = signed_angle(v_lower_sag, v_upper_sag, [1, 0, 0])
```

**Thresholds**:
- Neutral lordosis: 160° - 180°
- Hyperlordosis (excessive arch): < 150°
- Flexion (flattened): > 180°

### 4. Additional Metrics

**Knee Flexion Angle**: Sagittal plane knee bend (0° = extended)
**Hip Flexion Angle**: Angle between torso and thigh
**Squat Depth**: Normalized pelvis height (0 = standing, 1 = bottom)

---

## Squat Phase Detection

### Depth Normalization
```python
depth = (max_pelvis_height - current_height) / (max_height - min_height)
```

### Phase Identification
A squat repetition is detected when:
1. Depth crosses above 20% threshold (descending)
2. Maximum depth point is identified (bottom)
3. Depth crosses below threshold (ascending)
4. Minimum phase duration: 10 frames

### Clinical Focus
Metrics are primarily evaluated at the **bottom of squat** (±5 frames) where compensations are most apparent.

---

## Statistical Analysis

### Per-Subject Summary
For each metric, we compute:
- **Mean**: Average value during bottom-of-squat
- **Standard Deviation**: Movement consistency
- **Range**: [min, max] values observed
- **Percentiles**: 5th and 95th for outlier-robust assessment
- **Sample size**: Number of valid frames

### Compensation Flagging
Compensations are flagged when 95th percentile exceeds clinical thresholds, ensuring occasional peaks are captured.

---

## Output Format

### Markdown Report (`reports/results_report.md`)
- Assessment criteria and angle conventions
- Per-subject detailed metrics with tables
- Compensation pattern identification
- Clinical interpretation notes

### JSON Report (`reports/results_report.json`)
Machine-readable format for integration with other systems:
```json
{
  "generated": "2026-01-07T20:10:32",
  "subjects": [{
    "subject_id": "1",
    "classifications": {
      "left_knee": "neutral",
      "right_knee": "neutral",
      "torso": "neutral",
      "lumbar": "neutral"
    },
    "compensation_flags": ["Low back arches"]
  }]
}
```

---

## Validation

### Unit Tests
Comprehensive test suite covering:
- Vector operations (38 test cases)
- Angle calculations (perpendicular, parallel, signed angles)
- Filtering algorithms (spike removal, smoothing)
- Assessment metrics (knee, torso, lumbar)
- Squat phase detection

### Clinical Validation Considerations
- Results should be reviewed by qualified movement professionals
- Thresholds may need adjustment based on population norms
- Consider bilateral asymmetry in knee assessments
