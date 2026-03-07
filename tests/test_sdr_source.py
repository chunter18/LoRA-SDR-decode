"""Tests for sdr_source.py — CS16 IQ parsing (no hardware needed)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdr_source import parse_cs16, BYTES_PER_SAMPLE


class TestParseCS16:
    def test_zero_samples(self):
        """All-zero bytes produce all-zero complex output."""
        buf = bytes(BYTES_PER_SAMPLE * 10)
        result = parse_cs16(buf)
        assert len(result) == 10
        assert np.allclose(result, 0.0)

    def test_known_values(self):
        """Known int16 values produce expected complex output."""
        # I=1000, Q=2000 -> I/32768 + j*Q/32768
        i_val = np.int16(1000)
        q_val = np.int16(2000)
        buf = np.array([i_val, q_val], dtype=np.int16).tobytes()
        result = parse_cs16(buf)
        assert len(result) == 1
        expected = 1000 / 32768.0 + 1j * 2000 / 32768.0
        assert abs(result[0] - expected) < 1e-6

    def test_negative_values(self):
        """Negative int16 values produce negative float components."""
        i_val = np.int16(-16384)
        q_val = np.int16(-8192)
        buf = np.array([i_val, q_val], dtype=np.int16).tobytes()
        result = parse_cs16(buf)
        assert result[0].real < 0
        assert result[0].imag < 0
        assert abs(result[0].real - (-16384 / 32768.0)) < 1e-6
        assert abs(result[0].imag - (-8192 / 32768.0)) < 1e-6

    def test_full_scale(self):
        """Max int16 values normalize close to +/-1."""
        buf = np.array([32767, 32767], dtype=np.int16).tobytes()
        result = parse_cs16(buf)
        assert abs(result[0].real - 1.0) < 0.001
        assert abs(result[0].imag - 1.0) < 0.001

        buf = np.array([-32768, -32768], dtype=np.int16).tobytes()
        result = parse_cs16(buf)
        assert abs(result[0].real - (-1.0)) < 0.001
        assert abs(result[0].imag - (-1.0)) < 0.001

    def test_multiple_samples(self):
        """Multiple IQ pairs are parsed correctly."""
        data = np.array([100, 200, 300, 400, 500, 600], dtype=np.int16)
        buf = data.tobytes()
        result = parse_cs16(buf)
        assert len(result) == 3
        assert abs(result[0].real - 100 / 32768.0) < 1e-6
        assert abs(result[1].imag - 400 / 32768.0) < 1e-6
        assert abs(result[2].real - 500 / 32768.0) < 1e-6

    def test_output_dtype(self):
        """Output is complex64."""
        buf = bytes(BYTES_PER_SAMPLE * 4)
        result = parse_cs16(buf)
        assert result.dtype == np.complex64

    def test_invalid_length_raises(self):
        """Non-multiple-of-4 byte length raises ValueError."""
        with pytest.raises(ValueError):
            parse_cs16(b'\x00\x00\x00')  # 3 bytes, not multiple of 4
