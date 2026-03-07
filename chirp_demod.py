"""LoRa Symbol Demodulator — extract raw symbol values from IQ samples."""

import numpy as np
from chirp_detect import LoraParams, generate_chirp


def generate_lora_frame(params, n_preamble=8, sync_word=None, payload_symbols=None):
    """Generate a complete synthetic LoRa frame for testing.

    Frame structure: [preamble up-chirps] [sync word] [2.25 SFD down-chirps] [data]

    Args:
        params: LoraParams.
        n_preamble: Number of preamble up-chirps (default 8).
        sync_word: List of 2 symbol values for sync word (default [0, 0]).
        payload_symbols: List of data symbol values (0 to 2^SF - 1).

    Returns:
        Complex64 numpy array of the full frame.
    """
    if sync_word is None:
        sync_word = [0, 0]
    if payload_symbols is None:
        payload_symbols = []

    up = generate_chirp(params, direction='up')
    down = generate_chirp(params, direction='down')
    sym_len = params.symbol_samples

    parts = []

    # Preamble: unmodulated up-chirps
    parts.append(np.tile(up, n_preamble))

    # Sync word: 2 modulated up-chirps
    for sv in sync_word:
        shift = int(sv * sym_len / params.n_chips)
        parts.append(np.roll(up, -shift))

    # SFD: 2.25 down-chirps
    parts.append(np.tile(down, 2))
    parts.append(down[:sym_len // 4])

    # Data symbols: modulated up-chirps
    for sv in payload_symbols:
        shift = int(sv * sym_len / params.n_chips)
        parts.append(np.roll(up, -shift))

    return np.concatenate(parts).astype(np.complex64)
