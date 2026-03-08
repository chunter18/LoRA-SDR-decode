"""Tests for chirp_demod.py — symbol demodulation."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chirp_detect import LoraParams, generate_chirp
from chirp_demod import generate_lora_frame, find_sfd, extract_symbols, demodulate

# 250 kHz rate (decimated, 2x oversampling of 125 kHz BW)
PARAMS = LoraParams(sf=7, bw=125e3, sample_rate=250e3)


class TestGenerateLoraFrame:
    def test_frame_length(self):
        """Frame with 8 preamble + 2 sync + 2.25 SFD + 5 data = 17.25 symbols."""
        frame = generate_lora_frame(PARAMS, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=[10, 20, 30, 40, 50])
        expected = int(17.25 * PARAMS.symbol_samples)
        assert len(frame) == expected

    def test_preamble_region_is_up_chirps(self):
        """First 8 symbols should be unmodulated up-chirps (dechirp peak at bin 0)."""
        frame = generate_lora_frame(PARAMS, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=[10])
        ref_up = generate_chirp(PARAMS, direction='up')
        sym_len = PARAMS.symbol_samples
        for i in range(8):
            window = frame[i * sym_len:(i + 1) * sym_len]
            spectrum = np.abs(np.fft.fft(window * np.conj(ref_up)))
            assert np.argmax(spectrum) == 0

    def test_sfd_region_is_down_chirps(self):
        """Symbols 10-11 (after 8 preamble + 2 sync) should be down-chirps."""
        frame = generate_lora_frame(PARAMS, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=[10])
        ref_down = generate_chirp(PARAMS, direction='down')
        sym_len = PARAMS.symbol_samples
        # SFD starts at symbol index 10
        for i in [10, 11]:
            window = frame[i * sym_len:(i + 1) * sym_len]
            spectrum = np.abs(np.fft.fft(window * np.conj(ref_down)))
            assert np.argmax(spectrum) == 0


class TestFindSFD:
    def test_finds_sfd_in_clean_frame(self):
        """find_sfd returns correct sample offset of first data symbol."""
        params = PARAMS
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=[10, 20, 30])
        sym_len = params.symbol_samples
        preamble_end = 8 * sym_len
        data_offset = find_sfd(frame, params, preamble_end)
        # Data starts after 2 sync + 2.25 SFD = 4.25 symbols after preamble end
        expected = int((8 + 2 + 2.25) * sym_len)
        # Quarter-symbol scan step + boundary straddling = ~half symbol tolerance
        assert abs(data_offset - expected) <= sym_len // 2

    def test_finds_sfd_with_noise(self):
        """find_sfd works with moderate noise."""
        params = PARAMS
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=[10, 20])
        rng = np.random.default_rng(42)
        noise = 0.3 * (rng.standard_normal(len(frame)) +
                       1j * rng.standard_normal(len(frame)))
        noisy = (frame + noise).astype(np.complex64)

        sym_len = params.symbol_samples
        preamble_end = 8 * sym_len
        data_offset = find_sfd(noisy, params, preamble_end)
        expected = int((8 + 2 + 2.25) * sym_len)
        assert abs(data_offset - expected) <= sym_len // 2


class TestExtractSymbols:
    def test_extract_known_symbols(self):
        """Extract symbols from a clean frame with known payload."""
        params = PARAMS
        payload = [10, 20, 30, 40, 50]
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=payload)
        sym_len = params.symbol_samples
        data_offset = int((8 + 2 + 2.25) * sym_len)
        symbols = extract_symbols(frame, params, data_offset, n_symbols=5)
        assert symbols == payload

    def test_extract_with_noise(self):
        """Extract symbols from a noisy frame."""
        params = PARAMS
        payload = [10, 20, 30, 40, 50]
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=payload)
        rng = np.random.default_rng(42)
        noise = 0.3 * (rng.standard_normal(len(frame)) +
                       1j * rng.standard_normal(len(frame)))
        noisy = (frame + noise).astype(np.complex64)
        sym_len = params.symbol_samples
        data_offset = int((8 + 2 + 2.25) * sym_len)
        symbols = extract_symbols(noisy, params, data_offset, n_symbols=5)
        assert symbols == payload

    def test_extract_zero_symbols(self):
        """All-zero symbols return zeros."""
        params = PARAMS
        payload = [0, 0, 0]
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=payload)
        sym_len = params.symbol_samples
        data_offset = int((8 + 2 + 2.25) * sym_len)
        symbols = extract_symbols(frame, params, data_offset, n_symbols=3)
        assert symbols == [0, 0, 0]

    def test_extract_max_symbol(self):
        """Maximum symbol value (n_chips - 1) is extracted correctly."""
        params = PARAMS
        max_sym = params.n_chips - 1  # 127 for SF7
        payload = [max_sym]
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=payload)
        sym_len = params.symbol_samples
        data_offset = int((8 + 2 + 2.25) * sym_len)
        symbols = extract_symbols(frame, params, data_offset, n_symbols=1)
        assert symbols == [max_sym]


class TestDemodulate:
    def test_full_pipeline_clean(self):
        """demodulate() recovers known symbols from a clean frame at 2x rate."""
        params = PARAMS
        payload = [10, 20, 30, 40, 50]
        frame = generate_lora_frame(params, n_preamble=8, sync_word=[0, 0],
                                    payload_symbols=payload)
        # demodulate expects to decimate, but PARAMS is already at 2*BW,
        # so decimation factor=1 (passthrough).
        results = demodulate(frame, params, n_data_symbols=5)
        assert len(results) == 1
        assert results[0]['symbols'] == payload

    def test_full_pipeline_at_higher_sample_rate(self):
        """demodulate() works at 8x oversampling (1 MHz for 125 kHz BW)."""
        params_1m = LoraParams(sf=7, bw=125e3, sample_rate=1e6)
        payload = [10, 20, 30]
        frame = generate_lora_frame(params_1m, n_preamble=8,
                                    sync_word=[0, 0],
                                    payload_symbols=payload)
        results = demodulate(frame, params_1m, n_data_symbols=3)
        assert len(results) == 1
        assert results[0]['symbols'] == payload

    def test_no_detection_on_noise(self):
        """demodulate() returns empty list on pure noise."""
        params = PARAMS
        rng = np.random.default_rng(123)
        noise = (rng.standard_normal(10000) +
                 1j * rng.standard_normal(10000)).astype(np.complex64)
        results = demodulate(noise, params)
        assert results == []
