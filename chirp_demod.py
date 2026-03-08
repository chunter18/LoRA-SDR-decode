"""LoRa Symbol Demodulator — extract raw symbol values from IQ samples."""

import numpy as np
from scipy.signal import decimate as scipy_decimate
from chirp_detect import (LoraParams, generate_chirp, dechirp_and_fft,
                          detect_preamble)


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


def fine_align(samples, params, coarse_start, search_range=None):
    """Refine preamble timing by maximizing dechirped FFT peak.

    Searches around coarse_start for the offset that produces the
    sharpest dechirped peak (all chirp energy in one FFT bin).

    Args:
        samples: Complex IQ samples.
        params: LoraParams.
        coarse_start: Coarse preamble start (from detect_preamble).
        search_range: Samples to search ± around coarse_start.
                      Default: quarter-symbol.

    Returns:
        Refined sample offset (int).
    """
    sym_len = params.symbol_samples
    if search_range is None:
        search_range = sym_len // 4
    ref_up_conj = np.conj(generate_chirp(params, direction='up'))

    # Search step: ~1/32 of a symbol for good resolution
    step = max(1, sym_len // 32)
    lo = max(0, coarse_start - search_range)
    hi = min(len(samples) - sym_len, coarse_start + search_range)

    best_offset = coarse_start
    best_peak = 0.0
    for off in range(lo, hi + 1, step):
        window = samples[off:off + sym_len]
        spectrum = np.abs(np.fft.fft(window * ref_up_conj))
        peak = np.max(spectrum)
        if peak > best_peak:
            best_peak = peak
            best_offset = off

    return best_offset


def estimate_frequency_offset(samples, params, preamble_start, n_windows=4):
    """Estimate frequency offset from preamble windows.

    Preamble chirps have symbol value 0, so the dechirped peak bin
    directly gives the frequency offset in FFT bins.

    Args:
        samples: Complex IQ samples.
        params: LoraParams.
        preamble_start: Sample offset of preamble start.
        n_windows: Number of preamble windows to average.

    Returns:
        Frequency offset in FFT bins (float).
    """
    ref_up = generate_chirp(params, direction='up')
    sym_len = params.symbol_samples
    bins = []
    for i in range(n_windows):
        start = preamble_start + i * sym_len
        end = start + sym_len
        if end > len(samples):
            break
        window = samples[start:end]
        _, peak_bin, _ = dechirp_and_fft(window, ref_up)
        bins.append(peak_bin)
    if not bins:
        return 0.0
    # Use circular median to handle wrapping near 0/sym_len
    return float(np.median(bins))


def extract_symbols(samples, params, data_offset, n_symbols, freq_offset=0.0):
    """Extract raw symbol values from the data region.

    Args:
        samples: Complex IQ samples.
        params: LoraParams.
        data_offset: Sample offset where data symbols begin.
        n_symbols: Number of symbols to extract.
        freq_offset: Frequency offset in FFT bins (from preamble).

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
        # Subtract frequency offset, then map to symbol range
        corrected = (peak_bin - freq_offset) % sym_len
        symbol = int(round(corrected)) % params.n_chips
        symbols.append(symbol)

    return symbols


def demodulate(samples, params, n_data_symbols=20, min_chirps=4, snr_threshold=5.0):
    """Full demodulation pipeline: detect preamble, find SFD, extract symbols.

    Args:
        samples: Complex IQ samples at params.sample_rate.
        params: LoraParams.
        n_data_symbols: Max number of data symbols to extract.
        min_chirps: Minimum preamble chirps for detection.
        snr_threshold: SNR threshold for preamble detection.

    Returns:
        List of result dicts, each with:
          - 'symbols': list of raw symbol values
          - 'preamble_offset': sample offset of preamble
          - 'data_offset': sample offset of data region
          - 'n_chirps': number of preamble chirps detected
          - 'snr_db': average preamble SNR
    """
    detections = detect_preamble(samples, params,
                                 min_chirps=min_chirps,
                                 snr_threshold=snr_threshold)
    results = []
    sym_len = params.symbol_samples

    for det in detections:
        preamble_start = det['sample_offset']
        # Cap preamble length — detect_preamble may count sync/SFD/data
        # as chirps too (especially in high-SNR conditions).
        n_preamble = min(det['n_chirps'], 10)
        preamble_end = preamble_start + n_preamble * sym_len

        data_offset = find_sfd(samples, params, preamble_end)

        # Estimate frequency offset from preamble at the SAME sub-symbol
        # alignment as data_offset. This ensures the timing-dependent
        # frequency shift is consistent between offset estimation and
        # data extraction.
        # Walk back from data_offset into the preamble region.
        n_back = (data_offset - preamble_start) // sym_len
        preamble_at_data_align = data_offset - n_back * sym_len
        freq_offset = estimate_frequency_offset(samples, params,
                                                preamble_at_data_align,
                                                n_windows=min(4, n_back))

        max_possible = (len(samples) - data_offset) // sym_len
        n_to_extract = min(n_data_symbols, max_possible)
        if n_to_extract <= 0:
            continue

        symbols = extract_symbols(samples, params, data_offset,
                                  n_to_extract, freq_offset=freq_offset)
        results.append({
            'symbols': symbols,
            'preamble_offset': preamble_start,
            'data_offset': data_offset,
            'freq_offset': freq_offset,
            'n_chirps': det['n_chirps'],
            'snr_db': det['snr_db'],
        })

    return results
