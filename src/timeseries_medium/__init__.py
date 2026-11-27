"""
Time-series medium on the medium-agnostic core (issue #27)
==========================================================
The third medium built on ``src/core``, added without changing the core. The
comparable quantity is the **rate** of a periodic sensor signal (heart rate from
an ECG, step rate from an accelerometer, pulse rate from a PPG), in cycles per
minute, read from one window (8 s in Exp 053).

- ``SensorWindow``        the Medium: a 1-D window, ``context["fs"]`` required
- ``FftRate``             strongest spectral line in 30-240 per minute
- ``AcfRate``             highest autocorrelation peak at 0.25-2 s
- ``WaveletRate``         Morlet scalogram with harmonic summation
- ``PeakRate``            time-domain event counting (Pan-Tompkins style)
- ``RateComparator``      log-ratio distance, "agree" / "double" / "half" / "off",
                          confidence-weighted median as consensus
- ``make_timeseries_analyzer``  a ready-made ``core.Analyzer``

See src/experiments/exp053_timeseries_medium.py for the experiment.
"""

from .medium import (
    ALL_LENSES,
    LENSES,
    AcfRate,
    FftRate,
    PeakRate,
    RateComparator,
    SensorWindow,
    WaveletRate,
    is_gross,
    make_timeseries_analyzer,
    rate_error,
)

__all__ = [
    "ALL_LENSES",
    "LENSES",
    "AcfRate",
    "FftRate",
    "PeakRate",
    "RateComparator",
    "SensorWindow",
    "WaveletRate",
    "is_gross",
    "make_timeseries_analyzer",
    "rate_error",
]
