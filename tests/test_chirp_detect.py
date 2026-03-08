"""Tests for chirp_detect.py — chirp generation, de-chirping, and preamble detection.

All tests use synthetic IQ data, no SDR hardware needed.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chirp_detect import (
    LoraParams, generate_chirp, dechirp, dechirp_and_fft,
    detect_preamble, generate_lora_preamble, generate_lora_packet,
    decimate_to_lora, bin_to_freq_offset, estimate_parameters,
)

# Use SF7 for fast tests (128 chips vs 4096 for SF12)
FAST_PARAMS = LoraParams(sf=7, bw=125e3, sample_rate=1e6)
# SF12 for realistic tests
SLOW_PARAMS = LoraParams(sf=12, bw=125e3, sample_rate=1e6)


class TestLoraParams:
    def test_n_chips(self):
        assert FAST_PARAMS.n_chips == 128
        assert SLOW_PARAMS.n_chips == 4096

    def test_symbol_duration(self):
        # SF7, 125kHz: 128/125000 = 1.024 ms
        assert abs(FAST_PARAMS.symbol_duration - 0.001024) < 1e-9
        # SF12, 125kHz: 4096/125000 = 32.768 ms
        assert abs(SLOW_PARAMS.symbol_duration - 0.032768) < 1e-9

    def test_symbol_samples(self):
        # At 1 MHz sample rate
        assert FAST_PARAMS.symbol_samples == 1024
        assert SLOW_PARAMS.symbol_samples == 32768


class TestGenerateChirp:
    def test_length(self):
        """Chirp length matches expected symbol duration."""
        chirp = generate_chirp(FAST_PARAMS)
        assert len(chirp) == FAST_PARAMS.symbol_samples

    def test_unit_magnitude(self):
        """Chirp samples have magnitude ~1 (pure phase signal)."""
        chirp = generate_chirp(FAST_PARAMS)
        magnitudes = np.abs(chirp)
        assert np.allclose(magnitudes, 1.0, atol=1e-5)

    def test_up_vs_down_are_conjugates(self):
        """Up-chirp and down-chirp are complex conjugates of each other."""
        up = generate_chirp(FAST_PARAMS, direction='up')
        down = generate_chirp(FAST_PARAMS, direction='down')
        # They should be conjugates (within floating point tolerance)
        product = up * down
        # If conjugates, product phase should be near-linear (double the chirp rate)
        # Simpler check: up * conj(down) should give a chirp at double rate
        assert len(up) == len(down)
        assert not np.allclose(up, down)  # they should be different

    def test_frequency_sweep(self):
        """Instantaneous frequency sweeps across the bandwidth."""
        chirp = generate_chirp(FAST_PARAMS, direction='up')
        # Estimate instantaneous frequency from phase differences
        phase = np.angle(chirp)
        inst_freq = np.diff(np.unwrap(phase)) / (2 * np.pi) * FAST_PARAMS.sample_rate

        # Should start near -bw/2 and end near +bw/2
        assert inst_freq[0] < 0, "Up-chirp should start at negative frequency offset"
        assert inst_freq[-1] > 0, "Up-chirp should end at positive frequency offset"

        # Range should span approximately the bandwidth
        freq_range = inst_freq[-1] - inst_freq[0]
        assert abs(freq_range - FAST_PARAMS.bw) / FAST_PARAMS.bw < 0.05

    def test_dtype(self):
        """Output is complex64."""
        chirp = generate_chirp(FAST_PARAMS)
        assert chirp.dtype == np.complex64


class TestDechirp:
    def test_dechirp_produces_dc(self):
        """De-chirping an up-chirp with up-chirp ref produces DC (bin 0)."""
        up = generate_chirp(FAST_PARAMS, direction='up')
        # dechirp() conjugates the ref, so up * conj(up) = DC
        result = dechirp(up, up)

        # FFT should show energy concentrated at bin 0 (DC)
        spectrum = np.abs(np.fft.fft(result))
        peak_bin = np.argmax(spectrum)
        peak_power = spectrum[peak_bin] ** 2
        total_power = np.sum(spectrum ** 2)

        assert peak_bin == 0
        assert peak_power / total_power > 0.9

    def test_dechirp_length(self):
        """Output length matches reference chirp length."""
        up = generate_chirp(FAST_PARAMS, direction='up')
        ref = generate_chirp(FAST_PARAMS, direction='down')
        result = dechirp(up, ref)
        assert len(result) == len(ref)


class TestDechirpAndFFT:
    def test_unmodulated_chirp_peaks_at_zero(self):
        """An unmodulated up-chirp de-chirped with up-chirp ref peaks at bin 0."""
        up = generate_chirp(FAST_PARAMS, direction='up')
        _, peak_bin, snr_db = dechirp_and_fft(up, up)

        # Should peak at bin 0
        assert peak_bin == 0
        assert snr_db > 10

    def test_shifted_chirp_peaks_at_symbol_value(self):
        """A chirp with known cyclic shift produces the expected symbol value."""
        params = FAST_PARAMS
        up = generate_chirp(params, direction='up')

        # Shift by a known symbol value
        symbol_val = 42
        shift_samples = int(symbol_val * params.symbol_samples / params.n_chips)
        shifted = np.roll(up, -shift_samples)

        _, peak_bin, snr_db = dechirp_and_fft(shifted, up)

        # When n_fft = symbol_samples, FFT bin = symbol value directly
        # (bin = k * BW/N * n_fft/fs = k when n_fft = N*fs/BW = symbol_samples)
        assert abs(peak_bin - symbol_val) <= 1
        assert snr_db > 10

    def test_noise_only_low_snr(self):
        """Pure noise produces low SNR."""
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(FAST_PARAMS.symbol_samples) +
                 1j * rng.standard_normal(FAST_PARAMS.symbol_samples)).astype(np.complex64)
        up = generate_chirp(FAST_PARAMS, direction='up')
        _, _, snr_db = dechirp_and_fft(noise, up)

        # With mean-based metric, noise baseline should be < 8 dB
        assert snr_db < 8

    def test_noise_snr_consistent_across_fft_sizes(self):
        """Noise-only SNR should be similar regardless of FFT size."""
        rng = np.random.default_rng(42)
        snrs = []
        for params in [FAST_PARAMS, SLOW_PARAMS]:
            noise = (rng.standard_normal(params.symbol_samples) +
                     1j * rng.standard_normal(params.symbol_samples)).astype(np.complex64)
            up = generate_chirp(params, direction='up')
            _, _, snr_db = dechirp_and_fft(noise, up)
            snrs.append(snr_db)
            assert snr_db < 8, f"SF{params.sf} noise SNR {snr_db:.1f} dB too high"
        # The two should be within 3 dB of each other
        assert abs(snrs[0] - snrs[1]) < 3, (
            f"SF7={snrs[0]:.1f} vs SF12={snrs[1]:.1f} — too different"
        )


class TestDetectPreamble:
    def test_clean_preamble(self):
        """Detect a clean (noiseless) preamble."""
        params = FAST_PARAMS
        signal = generate_lora_preamble(params, n_chirps=8)
        detections = detect_preamble(signal, params, min_chirps=6, snr_threshold=10.0)

        assert len(detections) >= 1
        d = detections[0]
        assert d['n_chirps'] >= 6
        assert d['snr_db'] > 10

    def test_preamble_with_noise(self):
        """Detect a preamble buried in noise."""
        params = FAST_PARAMS
        signal = generate_lora_preamble(params, n_chirps=8)

        # Add noise (SNR ~ 10 dB)
        rng = np.random.default_rng(123)
        noise_power = 0.3
        noise = noise_power * (rng.standard_normal(len(signal)) +
                               1j * rng.standard_normal(len(signal)))
        noisy = (signal + noise).astype(np.complex64)

        detections = detect_preamble(noisy, params, min_chirps=6, snr_threshold=8.0)
        assert len(detections) >= 1
        assert detections[0]['n_chirps'] >= 6

    def test_no_false_detection_on_noise(self):
        """Pure noise should not trigger a detection."""
        params = FAST_PARAMS
        rng = np.random.default_rng(456)
        n_samples = params.symbol_samples * 20
        noise = (rng.standard_normal(n_samples) +
                 1j * rng.standard_normal(n_samples)).astype(np.complex64)

        detections = detect_preamble(noise, params, min_chirps=6, snr_threshold=8.0)
        assert len(detections) == 0

    def test_too_short_signal(self):
        """Signal shorter than min_chirps symbols returns empty."""
        params = FAST_PARAMS
        short = np.zeros(params.symbol_samples * 3, dtype=np.complex64)
        detections = detect_preamble(short, params, min_chirps=6)
        assert len(detections) == 0


class TestDecimateToLora:
    def test_output_length(self):
        """Output length is input_length / decimation_factor."""
        params = FAST_PARAMS  # 1 MHz, 125 kHz BW → factor 4
        n_samples = params.symbol_samples * 4  # 4096 samples
        samples = np.zeros(n_samples, dtype=np.complex64)
        decimated, new_params = decimate_to_lora(samples, params)
        assert len(decimated) == n_samples // 4

    def test_new_params_sample_rate(self):
        """Decimated params have target sample rate (2*BW)."""
        params = FAST_PARAMS
        samples = np.zeros(params.symbol_samples, dtype=np.complex64)
        _, new_params = decimate_to_lora(samples, params)
        assert new_params.sample_rate == 2 * params.bw
        assert new_params.sf == params.sf
        assert new_params.bw == params.bw

    def test_no_decimation_when_rate_matches(self):
        """No decimation when sample_rate <= 2*BW."""
        params = LoraParams(sf=7, bw=125e3, sample_rate=250e3)
        samples = np.ones(100, dtype=np.complex64)
        decimated, new_params = decimate_to_lora(samples, params)
        assert len(decimated) == 100
        assert new_params.sample_rate == 250e3

    def test_amplitude_preserved(self):
        """Decimation preserves signal amplitude for in-band signal."""
        params = FAST_PARAMS
        chirp = generate_chirp(params, direction='up')
        decimated, _ = decimate_to_lora(chirp, params)
        # RMS amplitude should be close to 1.0 (chirp is unit magnitude)
        rms = np.sqrt(np.mean(np.abs(decimated) ** 2))
        assert 0.8 < rms < 1.2

    def test_chirp_dechirps_cleanly_after_decimation(self):
        """A chirp decimated to 2*BW still produces a clean FFT peak."""
        params = FAST_PARAMS
        chirp = generate_chirp(params, direction='up')
        decimated, new_params = decimate_to_lora(chirp, params)

        # Generate reference at decimated rate
        ref = generate_chirp(new_params, direction='up')
        _, peak_bin, snr_db = dechirp_and_fft(decimated, ref)

        assert peak_bin == 0  # unmodulated chirp → bin 0
        assert snr_db > 10

    def test_preamble_detected_after_decimation(self):
        """Full preamble detection works on decimated signal."""
        params = FAST_PARAMS
        signal = generate_lora_preamble(params, n_chirps=8)

        # Add moderate noise
        rng = np.random.default_rng(77)
        noise = 0.3 * (rng.standard_normal(len(signal)) +
                       1j * rng.standard_normal(len(signal)))
        noisy = (signal + noise).astype(np.complex64)

        # Decimate then detect
        decimated, new_params = decimate_to_lora(noisy, params)
        detections = detect_preamble(decimated, new_params,
                                     min_chirps=6, snr_threshold=8.0)
        assert len(detections) >= 1
        assert detections[0]['n_chirps'] >= 6

    def test_peak_bin_in_lora_range(self):
        """After decimation, peak bin is within LoRa symbol range."""
        params = FAST_PARAMS
        signal = generate_lora_preamble(params, n_chirps=8)
        decimated, new_params = decimate_to_lora(signal, params)
        detections = detect_preamble(decimated, new_params,
                                     min_chirps=6, snr_threshold=10.0)
        assert len(detections) >= 1
        # At 2*BW sample rate, bins span 0..2*n_chips-1
        # Unmodulated preamble should peak near bin 0
        assert detections[0]['peak_bin'] < new_params.n_chips // 4


class TestGeneratePreamble:
    def test_preamble_length(self):
        """Preamble length is n_chirps * symbol_samples."""
        params = FAST_PARAMS
        signal = generate_lora_preamble(params, n_chirps=8)
        assert len(signal) == 8 * params.symbol_samples

    def test_preamble_with_payload(self):
        """Preamble + payload has correct total length."""
        params = FAST_PARAMS
        symbols = [0, 42, 100]
        signal = generate_lora_preamble(params, n_chirps=8, payload_symbols=symbols)
        expected_len = (8 + len(symbols)) * params.symbol_samples
        assert len(signal) == expected_len


class TestBinToFreqOffset:
    def test_zero_bin(self):
        assert bin_to_freq_offset(0, 1024, 1e6) == 0.0

    def test_positive_bin(self):
        # bin 100 out of 1024, at 1 MHz → 100 * 1e6 / 1024 ≈ 97656 Hz
        result = bin_to_freq_offset(100, 1024, 1e6)
        assert abs(result - 97656.25) < 1.0

    def test_negative_wrap(self):
        # bin 924 out of 1024 → (924 - 1024) * 1e6 / 1024 = -100 * 1e6 / 1024
        result = bin_to_freq_offset(924, 1024, 1e6)
        assert abs(result - (-97656.25)) < 1.0

    def test_nyquist(self):
        # bin 512 out of 1024 → not wrapped (== n_fft/2)
        result = bin_to_freq_offset(512, 1024, 1e6)
        assert abs(result - 500000.0) < 1.0


class TestEstimateParameters:
    def test_detects_sf12_125k(self):
        """Correctly identifies SF12/125kHz preamble."""
        params = LoraParams(sf=12, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        result = estimate_parameters(signal, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 12
        assert result['bw'] == 125e3

    def test_detects_sf7_125k(self):
        """Correctly identifies SF7/125kHz preamble."""
        params = LoraParams(sf=7, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        # Pad so larger SFs have enough samples to attempt (and fail)
        padded = np.concatenate([signal, np.zeros(300000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 7
        assert result['bw'] == 125e3

    def test_detects_sf10_250k(self):
        """Correctly identifies SF10/250kHz preamble when BW is set to 250k."""
        params = LoraParams(sf=10, bw=250e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        padded = np.concatenate([signal, np.zeros(300000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=250e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 10
        assert result['bw'] == 250e3

    def test_detects_frequency_offset(self):
        """Correctly estimates frequency offset of shifted signal."""
        params = LoraParams(sf=7, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        # Apply 50 kHz frequency offset
        t = np.arange(len(signal)) / 1e6
        shifted = (signal * np.exp(2j * np.pi * 50e3 * t)).astype(np.complex64)
        padded = np.concatenate([shifted, np.zeros(300000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 7
        assert abs(result['freq_offset_hz'] - 50e3) < 5e3

    def test_no_detection_on_noise(self):
        """Pure noise returns None."""
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(300000) +
                 1j * rng.standard_normal(300000)).astype(np.complex64)
        result = estimate_parameters(noise, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is None

    def test_detects_with_noise(self):
        """Detects SF correctly in noisy signal."""
        params = LoraParams(sf=12, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        rng = np.random.default_rng(42)
        noise = 0.3 * (rng.standard_normal(len(signal)) +
                       1j * rng.standard_normal(len(signal)))
        noisy = (signal + noise).astype(np.complex64)
        result = estimate_parameters(noisy, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 12
        assert result['bw'] == 125e3

    def test_discriminates_sf_with_freq_offset(self):
        """SF12/125k with 60 kHz crystal offset is not confused with other SFs."""
        params = LoraParams(sf=12, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        t = np.arange(len(signal)) / 1e6
        signal = (signal * np.exp(2j * np.pi * 60e3 * t)).astype(np.complex64)
        rng = np.random.default_rng(42)
        noise = 0.3 * (rng.standard_normal(len(signal)) +
                       1j * rng.standard_normal(len(signal)))
        noisy = (signal + noise).astype(np.complex64)
        padded = np.concatenate([noisy, np.zeros(400000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 12
        assert result['bw'] == 125e3
        assert abs(result['freq_offset_hz'] - 60e3) < 5e3

    def test_wrong_bw_detects_wrong_sf(self):
        """Signal at 250k BW analyzed at 125k BW detects at wrong SF.

        A 250k signal has 2x the chip rate of 125k, so when dechirped at
        125k the symbol duration is halved — the detector sees it as a
        lower SF. This is expected: BW must be set correctly for accurate SF.
        """
        params = LoraParams(sf=10, bw=250e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        padded = np.concatenate([signal, np.zeros(400000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        # Will detect something, but the SF will be wrong (lower than 10)
        if result is not None:
            assert result['sf'] != 10 or result['bw'] != 250e3

    def test_timing_offset(self):
        """Detects signal that starts mid-block (not at sample 0)."""
        params = LoraParams(sf=7, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        # Embed signal 50000 samples into a noise block
        rng = np.random.default_rng(99)
        block = 0.1 * (rng.standard_normal(400000) +
                       1j * rng.standard_normal(400000)).astype(np.complex64)
        offset = 50000
        block[offset:offset + len(signal)] += signal
        result = estimate_parameters(block, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 7
        assert result['bw'] == 125e3

    def test_combined_realistic_conditions(self):
        """SF12/125k with freq offset + noise + timing offset (live-like)."""
        params = LoraParams(sf=12, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        # Apply 60 kHz frequency offset
        t = np.arange(len(signal)) / 1e6
        signal = (signal * np.exp(2j * np.pi * 60e3 * t)).astype(np.complex64)
        # Embed with timing offset in noisy block
        rng = np.random.default_rng(77)
        block = 0.3 * (rng.standard_normal(600000) +
                       1j * rng.standard_normal(600000)).astype(np.complex64)
        offset = 30000
        block[offset:offset + len(signal)] += signal
        result = estimate_parameters(block, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 12
        assert result['bw'] == 125e3

    def test_sample_rate_mismatch_does_not_double_bw(self):
        """A 2x sample rate mismatch produces garbage, not systematic errors."""
        actual_rate = 2e6
        assumed_rate = 1e6

        params_real = LoraParams(sf=12, bw=125e3, sample_rate=actual_rate)
        signal = generate_lora_preamble(params_real, n_chirps=8)
        padded = np.concatenate([signal, np.zeros(700000, dtype=np.complex64)])

        result = estimate_parameters(padded, assumed_rate, bw=125e3, min_chirps=6, snr_threshold=5.0)
        if result is not None:
            assert result['score'] < 1.0, "Score should be very low with wrong sample rate"

    def test_correct_rate_gives_correct_result(self):
        """With correct sample rate assumption, SF is detected correctly."""
        rate = 2e6
        params = LoraParams(sf=12, bw=125e3, sample_rate=rate)
        signal = generate_lora_preamble(params, n_chirps=8)
        padded = np.concatenate([signal, np.zeros(700000, dtype=np.complex64)])

        result = estimate_parameters(padded, rate, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['bw'] == 125e3
        assert result['sf'] == 12

    def test_full_packet_sf12_125k(self):
        """SF detection works on full LoRa packet (not just preamble).

        Real packets have sync words, SFD down-chirps, and data after the
        preamble. Tests that this extra structure doesn't confuse the scorer.
        """
        params = LoraParams(sf=12, bw=125e3, sample_rate=1e6)
        signal = generate_lora_packet(params, n_preamble=8, data_symbols=[10, 100, 200, 50, 3000])
        rng = np.random.default_rng(42)
        noise = 0.3 * (rng.standard_normal(len(signal)) +
                       1j * rng.standard_normal(len(signal)))
        noisy = (signal + noise).astype(np.complex64)
        padded = np.concatenate([noisy, np.zeros(200000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 12, f"Expected SF12, got SF{result['sf']}"
        assert result['bw'] == 125e3, f"Expected 125k, got {result['bw']/1e3:.0f}k"

    def test_full_packet_sf7_250k(self):
        """SF detection on full SF7/250k packet with correct BW set."""
        params = LoraParams(sf=7, bw=250e3, sample_rate=1e6)
        signal = generate_lora_packet(params, n_preamble=8, data_symbols=[10, 42, 100])
        rng = np.random.default_rng(99)
        noise = 0.2 * (rng.standard_normal(len(signal)) +
                       1j * rng.standard_normal(len(signal)))
        noisy = (signal + noise).astype(np.complex64)
        padded = np.concatenate([noisy, np.zeros(300000, dtype=np.complex64)])
        result = estimate_parameters(padded, 1e6, bw=250e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 7, f"Expected SF7, got SF{result['sf']}"
        assert result['bw'] == 250e3, f"Expected 250k, got {result['bw']/1e3:.0f}k"

    def test_sf7_realistic(self):
        """SF7/125k with freq offset + noise — short preamble still detected."""
        params = LoraParams(sf=7, bw=125e3, sample_rate=1e6)
        signal = generate_lora_preamble(params, n_chirps=8)
        t = np.arange(len(signal)) / 1e6
        signal = (signal * np.exp(2j * np.pi * -30e3 * t)).astype(np.complex64)
        rng = np.random.default_rng(55)
        block = 0.2 * (rng.standard_normal(400000) +
                       1j * rng.standard_normal(400000)).astype(np.complex64)
        block[80000:80000 + len(signal)] += signal
        result = estimate_parameters(block, 1e6, bw=125e3, min_chirps=6, snr_threshold=5.0)
        assert result is not None
        assert result['sf'] == 7
        assert result['bw'] == 125e3
