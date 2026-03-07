"""LoRa Symbol Demodulator — extract raw symbol values from IQ samples."""

import numpy as np
from chirp_detect import LoraParams, generate_chirp, dechirp_and_fft


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


def find_sfd(samples, params, preamble_end):
    """Find the SFD (down-chirps) after a detected preamble.

    Scans forward from preamble_end looking for down-chirp energy.
    Returns the sample offset where data symbols begin (after the 2.25 SFD).

    Args:
        samples: Complex IQ samples.
        params: LoraParams.
        preamble_end: Sample offset of preamble end.

    Returns:
        Sample offset of the first data symbol.
    """
    ref_up_conj = np.conj(generate_chirp(params, direction='up'))
    ref_down_conj = np.conj(generate_chirp(params, direction='down'))
    sym_len = params.symbol_samples

    # Scan from preamble_end in quarter-symbol steps.
    # Expect: 2 sync up-chirps, then 2+ down-chirps.
    search_end = min(preamble_end + 6 * sym_len, len(samples) - sym_len)

    first_down = None
    offset = preamble_end
    while offset + sym_len <= search_end:
        window = samples[offset:offset + sym_len]
        up_peak = np.max(np.abs(np.fft.fft(window * ref_up_conj)))
        down_peak = np.max(np.abs(np.fft.fft(window * ref_down_conj)))

        if down_peak > up_peak:
            if first_down is None:
                first_down = offset
        elif first_down is not None:
            # Transitioned out of down-chirp region
            break

        offset += sym_len // 4

    if first_down is None:
        # Fallback: assume standard 2 sync + 2.25 SFD
        return preamble_end + int(4.25 * sym_len)

    # Data starts 2.25 down-chirp durations after the first down-chirp
    return first_down + int(2.25 * sym_len)


def extract_symbols(samples, params, data_offset, n_symbols):
    """Extract raw symbol values from the data region.

    Args:
        samples: Complex IQ samples.
        params: LoraParams.
        data_offset: Sample offset where data symbols begin.
        n_symbols: Number of symbols to extract.

    Returns:
        List of integer symbol values (0 to 2^SF - 1).
    """
    ref_up = generate_chirp(params, direction='up')
    sym_len = params.symbol_samples
    symbols = []

    for i in range(n_symbols):
        start = data_offset + i * sym_len
        end = start + sym_len
        if end > len(samples):
            break
        window = samples[start:end]
        _, peak_bin, _ = dechirp_and_fft(window, ref_up)
        symbol = peak_bin % params.n_chips
        symbols.append(symbol)

    return symbols
