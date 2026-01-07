"""
Signal Filtering Module for Motion Capture Data

This module provides noise filtering algorithms specifically designed for
3D motion capture data with potential sensor noise and outliers.

Implements a multi-stage filtering pipeline:
1. Median filter: Removes spike noise while preserving motion edges
2. Exponential Moving Average: Smooths high-frequency jitter
3. (Optional) Butterworth low-pass filter: Biomechanically-appropriate smoothing

References:
    - Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
    - Robertson, D.G.E. et al. (2013). Research Methods in Biomechanics.
"""

from typing import List, Optional, Tuple
import math


def median_filter(values: List[Optional[float]], window: int = 5) -> List[Optional[float]]:
    """
    Apply a median filter to remove spike noise from the signal.
    
    The median filter is particularly effective for motion capture data as it:
    - Removes impulsive noise (sensor glitches) without distorting edges
    - Preserves the timing of movement transitions
    - Is robust to outliers
    
    Args:
        values: Input signal with possible None values (missing data)
        window: Window size (must be odd, >= 3). Default 5 is suitable for 
                30-60 Hz capture rates.
    
    Returns:
        Filtered signal with same length as input
    
    Raises:
        ValueError: If window is not odd or < 3
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    
    radius = window // 2
    n = len(values)
    out = []
    
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        
        # Collect valid (non-None) values in window
        window_vals = [v for v in values[start:end] if v is not None]
        
        if not window_vals:
            out.append(None)
            continue
        
        # Compute median
        window_vals.sort()
        median_idx = len(window_vals) // 2
        out.append(window_vals[median_idx])
    
    return out


def ema_filter(
    values: List[Optional[float]], 
    alpha: float = 0.2
) -> List[Optional[float]]:
    """
    Apply Exponential Moving Average (EMA) filter for smoothing.
    
    EMA provides smooth transitions while maintaining responsiveness to
    real motion changes. The alpha parameter controls the trade-off:
    - Higher alpha (e.g., 0.5): More responsive, less smoothing
    - Lower alpha (e.g., 0.1): More smoothing, slower response
    
    Formula: y[n] = α * x[n] + (1-α) * y[n-1]
    
    Args:
        values: Input signal with possible None values
        alpha: Smoothing factor (0 < alpha <= 1). Default 0.2 is appropriate
               for human movement at typical capture rates.
    
    Returns:
        Smoothed signal
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    
    out = []
    prev = None
    
    for v in values:
        if v is None:
            # Propagate last valid value for continuity
            out.append(prev)
            continue
        
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        
        out.append(prev)
    
    return out


def bidirectional_ema(
    values: List[Optional[float]], 
    alpha: float = 0.2
) -> List[Optional[float]]:
    """
    Apply bidirectional (zero-phase) EMA filter.
    
    This eliminates phase lag by filtering forwards then backwards,
    which is important for preserving timing of movement events.
    
    Args:
        values: Input signal
        alpha: Smoothing factor
    
    Returns:
        Zero-phase filtered signal
    """
    # Forward pass
    forward = ema_filter(values, alpha)
    
    # Backward pass
    backward = ema_filter(forward[::-1], alpha)
    
    return backward[::-1]


def butterworth_lowpass_coefficients(
    cutoff_freq: float,
    sample_freq: float,
    order: int = 2
) -> Tuple[List[float], List[float]]:
    """
    Compute Butterworth low-pass filter coefficients.
    
    For biomechanical data, typical parameters:
    - Cutoff: 6-10 Hz for general movement
    - Cutoff: 10-15 Hz for faster movements
    - Order: 2 (second-order, commonly used)
    
    Args:
        cutoff_freq: Cutoff frequency in Hz
        sample_freq: Sampling frequency in Hz
        order: Filter order (2 is standard for biomechanics)
    
    Returns:
        Tuple of (b, a) filter coefficients
    """
    # Normalized cutoff frequency
    wc = math.tan(math.pi * cutoff_freq / sample_freq)
    
    if order == 2:
        # Second-order Butterworth
        k1 = math.sqrt(2) * wc
        k2 = wc * wc
        
        a0 = 1 + k1 + k2
        a1 = 2 * (k2 - 1) / a0
        a2 = (1 - k1 + k2) / a0
        
        b0 = k2 / a0
        b1 = 2 * k2 / a0
        b2 = k2 / a0
        
        return [b0, b1, b2], [1.0, a1, a2]
    else:
        raise ValueError("Only order=2 is currently supported")


def butterworth_filter(
    values: List[Optional[float]],
    cutoff_freq: float = 6.0,
    sample_freq: float = 30.0,
    order: int = 2
) -> List[Optional[float]]:
    """
    Apply Butterworth low-pass filter (zero-phase, forward-backward).
    
    This is the standard filter for biomechanical signal processing,
    providing optimal frequency response with no phase distortion.
    
    Args:
        values: Input signal
        cutoff_freq: Cutoff frequency in Hz (default 6 Hz for movement)
        sample_freq: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered signal
    """
    # Get filter coefficients
    b, a = butterworth_lowpass_coefficients(cutoff_freq, sample_freq, order)
    
    # Handle None values by interpolating
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    if len(valid_indices) < 3:
        return values  # Not enough data to filter
    
    # Create continuous signal with interpolation
    continuous = list(values)
    for i in range(len(continuous)):
        if continuous[i] is None:
            # Find nearest valid values
            prev_idx = max([j for j in valid_indices if j < i], default=valid_indices[0])
            next_idx = min([j for j in valid_indices if j > i], default=valid_indices[-1])
            
            if prev_idx == next_idx:
                continuous[i] = values[prev_idx]
            else:
                # Linear interpolation
                t = (i - prev_idx) / (next_idx - prev_idx)
                continuous[i] = values[prev_idx] + t * (values[next_idx] - values[prev_idx])
    
    n = len(continuous)
    
    # Forward filtering
    forward = [0.0] * n
    for i in range(n):
        if i == 0:
            forward[i] = b[0] * continuous[i]
        elif i == 1:
            forward[i] = (b[0] * continuous[i] + b[1] * continuous[i-1] 
                         - a[1] * forward[i-1])
        else:
            forward[i] = (b[0] * continuous[i] + b[1] * continuous[i-1] + b[2] * continuous[i-2]
                         - a[1] * forward[i-1] - a[2] * forward[i-2])
    
    # Backward filtering (for zero-phase)
    backward = [0.0] * n
    for i in range(n-1, -1, -1):
        if i == n-1:
            backward[i] = b[0] * forward[i]
        elif i == n-2:
            backward[i] = (b[0] * forward[i] + b[1] * forward[i+1]
                          - a[1] * backward[i+1])
        else:
            backward[i] = (b[0] * forward[i] + b[1] * forward[i+1] + b[2] * forward[i+2]
                          - a[1] * backward[i+1] - a[2] * backward[i+2])
    
    # Restore None values
    result = []
    for i, v in enumerate(values):
        if v is None:
            result.append(None)
        else:
            result.append(backward[i])
    
    return result


def smooth_series(
    values: List[Optional[float]], 
    window: int = 5, 
    alpha: float = 0.2,
    use_butterworth: bool = False,
    cutoff_freq: float = 6.0,
    sample_freq: float = 30.0
) -> List[Optional[float]]:
    """
    Apply multi-stage filtering pipeline for motion capture data.
    
    Default pipeline:
    1. Median filter (removes spikes)
    2. EMA filter (smooths noise)
    
    With use_butterworth=True:
    1. Median filter (removes spikes)  
    2. Butterworth low-pass filter (biomechanically appropriate smoothing)
    
    Args:
        values: Input signal
        window: Median filter window size
        alpha: EMA smoothing factor (ignored if use_butterworth=True)
        use_butterworth: Use Butterworth filter instead of EMA
        cutoff_freq: Butterworth cutoff frequency (Hz)
        sample_freq: Sampling frequency (Hz)
    
    Returns:
        Filtered signal
    """
    # Stage 1: Median filter for spike removal
    median_filtered = median_filter(values, window=window)
    
    # Stage 2: Smoothing
    if use_butterworth:
        return butterworth_filter(
            median_filtered, 
            cutoff_freq=cutoff_freq, 
            sample_freq=sample_freq
        )
    else:
        return ema_filter(median_filtered, alpha=alpha)


def detect_outliers_zscore(
    values: List[Optional[float]],
    threshold: float = 3.0
) -> List[bool]:
    """
    Detect outliers using Z-score method.
    
    Args:
        values: Input signal
        threshold: Z-score threshold for outlier detection
    
    Returns:
        Boolean list indicating outlier positions
    """
    valid_vals = [v for v in values if v is not None]
    if len(valid_vals) < 2:
        return [False] * len(values)
    
    mean = sum(valid_vals) / len(valid_vals)
    variance = sum((v - mean) ** 2 for v in valid_vals) / len(valid_vals)
    std = math.sqrt(variance) if variance > 0 else 1.0
    
    outliers = []
    for v in values:
        if v is None:
            outliers.append(False)
        else:
            z = abs(v - mean) / std
            outliers.append(z > threshold)
    
    return outliers


def interpolate_missing(
    values: List[Optional[float]],
    max_gap: int = 5
) -> List[Optional[float]]:
    """
    Interpolate missing (None) values using linear interpolation.
    
    Only interpolates gaps smaller than max_gap frames to avoid
    introducing artifacts from long missing segments.
    
    Args:
        values: Input signal with None values
        max_gap: Maximum gap length to interpolate
    
    Returns:
        Signal with small gaps filled
    """
    result = list(values)
    n = len(result)
    
    i = 0
    while i < n:
        if result[i] is None:
            # Find gap boundaries
            gap_start = i
            while i < n and result[i] is None:
                i += 1
            gap_end = i
            gap_length = gap_end - gap_start
            
            # Only interpolate short gaps
            if gap_length <= max_gap:
                # Find boundary values
                prev_val = result[gap_start - 1] if gap_start > 0 else None
                next_val = result[gap_end] if gap_end < n else None
                
                if prev_val is not None and next_val is not None:
                    # Linear interpolation
                    for j in range(gap_start, gap_end):
                        t = (j - gap_start + 1) / (gap_length + 1)
                        result[j] = prev_val + t * (next_val - prev_val)
                elif prev_val is not None:
                    # Forward fill
                    for j in range(gap_start, gap_end):
                        result[j] = prev_val
                elif next_val is not None:
                    # Backward fill
                    for j in range(gap_start, gap_end):
                        result[j] = next_val
        else:
            i += 1
    
    return result
