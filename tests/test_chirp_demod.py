"""Tests for chirp_demod.py — symbol demodulation."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chirp_detect import LoraParams, generate_chirp
from chirp_demod import generate_lora_frame

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
