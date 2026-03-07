"""Tests for chirp_monitor.py — deduplication and noise floor estimation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chirp_detect import LoraParams
from chirp_monitor import Deduplicator, estimate_noise_floor_dbfs


PARAMS_SF12 = LoraParams(sf=12, bw=125e3, sample_rate=1e6)
PARAMS_SF7 = LoraParams(sf=7, bw=125e3, sample_rate=1e6)


class TestDeduplicator:
    def test_first_detection_always_passes(self):
        """The very first detection for a given SF should always be reported."""
        dedup = Deduplicator()
        assert dedup.should_report(sf=12, timestamp=1.0, preamble_duration=0.262) is True

    def test_duplicate_within_preamble_duration_suppressed(self):
        """A detection within one preamble duration of the last should be suppressed."""
        dedup = Deduplicator()
        dedup.should_report(sf=12, timestamp=1.0, preamble_duration=0.262)
        # 0.1s later — well within 0.262s preamble duration
        assert dedup.should_report(sf=12, timestamp=1.1, preamble_duration=0.262) is False

    def test_detection_after_preamble_duration_passes(self):
        """A detection after the preamble duration has elapsed should be reported."""
        dedup = Deduplicator()
        dedup.should_report(sf=12, timestamp=1.0, preamble_duration=0.262)
        # 0.5s later — past the 0.262s window
        assert dedup.should_report(sf=12, timestamp=1.5, preamble_duration=0.262) is True

    def test_different_sf_independent(self):
        """Detections on different SFs should not suppress each other."""
        dedup = Deduplicator()
        dedup.should_report(sf=12, timestamp=1.0, preamble_duration=0.262)
        # Same time, different SF — should pass
        assert dedup.should_report(sf=7, timestamp=1.0, preamble_duration=0.001) is True

    def test_rapid_fire_only_first_passes(self):
        """Multiple rapid detections should only report the first."""
        dedup = Deduplicator()
        results = []
        for i in range(5):
            t = 1.0 + i * 0.05  # 50ms apart
            results.append(dedup.should_report(sf=12, timestamp=t, preamble_duration=0.262))
        assert results == [True, False, False, False, False]

    def test_periodic_detections_all_pass(self):
        """Detections spaced far apart should all be reported."""
        dedup = Deduplicator()
        results = []
        for i in range(5):
            t = i * 3.0  # 3 seconds apart (beacon interval)
            results.append(dedup.should_report(sf=12, timestamp=t, preamble_duration=0.262))
        assert results == [True, True, True, True, True]


class TestEstimateNoiseFloor:
    def test_unit_noise_near_expected(self):
        """Unit-variance complex noise should give a noise floor near a known level."""
        rng = np.random.default_rng(42)
        # Unit-variance complex noise: each component std=1/sqrt(2), so |z|^2 ~ 1
        samples = (rng.standard_normal(100000) +
                   1j * rng.standard_normal(100000)).astype(np.complex64)
        noise_dbfs = estimate_noise_floor_dbfs(samples)
        # Mean power of unit complex noise = 2.0 (real^2 + imag^2, each var=1)
        # 10*log10(2) ~ 3.0 dBFS
        # Allow some tolerance
        assert -1.0 < noise_dbfs < 7.0

    def test_quiet_signal_lower_than_noisy(self):
        """A quieter signal should have a lower noise floor than a louder one."""
        rng = np.random.default_rng(99)
        quiet = 0.01 * (rng.standard_normal(10000) +
                        1j * rng.standard_normal(10000)).astype(np.complex64)
        loud = 1.0 * (rng.standard_normal(10000) +
                      1j * rng.standard_normal(10000)).astype(np.complex64)
        assert estimate_noise_floor_dbfs(quiet) < estimate_noise_floor_dbfs(loud)

    def test_returns_finite(self):
        """Should return a finite float, not inf or nan."""
        rng = np.random.default_rng(7)
        samples = (rng.standard_normal(1000) +
                   1j * rng.standard_normal(1000)).astype(np.complex64)
        result = estimate_noise_floor_dbfs(samples)
        assert np.isfinite(result)
