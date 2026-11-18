"""Held-out residual intervals for fixed one-step forecasts.

Temporal residuals need not be exchangeable: nominal coverage is a target, not a guarantee.
"""
import math
import numpy as np


def calibrate(actual, predicted, coverage=0.9):
    """Calibrate a symmetric radius from a pre-test validation window.

    Use the finite-sample conformal order statistic ceil((n+1)*coverage),
    without interpolation. Reject requests whose rank exceeds the sample size.
    """
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError('coverage must be finite and strictly between zero and one')
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or not len(actual) or len(actual) != len(predicted):
        raise ValueError('actual and predicted must be nonempty equal-length vectors')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('actual and predicted must be finite')
    rank = math.ceil((len(actual) + 1) * coverage)
    if rank > len(actual):
        raise ValueError('calibration sample is too small for the requested coverage')
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = np.abs(actual - predicted)
    if not np.isfinite(residuals).all():
        raise ValueError('calibration residuals must be finite')
    radius = float(np.partition(residuals, rank - 1)[rank - 1])
    return {'target_coverage': float(coverage), 'radius': radius, 'calibration_count': len(actual), 'order_statistic': rank}


def evaluate_intervals(actual, predicted, calibration):
    """Measure inclusive empirical coverage and mean width on later observations."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or not len(actual) or len(actual) != len(predicted):
        raise ValueError('actual and predicted must be nonempty equal-length vectors')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('actual and predicted must be finite')
    radius = calibration['radius']
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or not math.isfinite(radius) or radius < 0:
        raise ValueError('radius must be finite and nonnegative')
    with np.errstate(over='ignore', invalid='ignore'):
        residuals = np.abs(actual - predicted)
    width = 2 * radius
    if not np.isfinite(residuals).all() or not math.isfinite(width):
        raise ValueError('interval residuals and width must be finite')
    coverage = float(np.mean(residuals <= radius))
    return {'count': len(actual), 'empirical_coverage': coverage,
            'target_coverage': calibration['target_coverage'],
            'coverage_gap': coverage - calibration['target_coverage'], 'mean_width': width}
