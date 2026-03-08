"""
LoRa Chirp Detector — de-chirp + FFT approach for preamble detection.

LoRa uses Chirp Spread Spectrum (CSS). Each symbol is a frequency chirp
sweeping across the bandwidth. Data is encoded in the chirp's starting
frequency. The preamble is 8+ identical unmodulated up-chirps.

Detection approach:
  1. Generate a reference down-chirp (conjugate of up-chirp)
  2. Multiply received signal by reference (de-chirping)
  3. FFT the result — a chirp collapses to a single peak
  4. Look for repeated peaks at consistent intervals (preamble)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class LoraParams:
    """LoRa modulation parameters."""
    sf: int = 12              # spreading factor (7-12)
    bw: float = 125e3         # bandwidth in Hz
    sample_rate: float = 1e6  # SDR sample rate in Hz

    @property
    def n_chips(self):
        """Number of chips per symbol (2^SF)."""
        return 1 << self.sf

    @property
    def symbol_duration(self):
        """Duration of one symbol in seconds."""
        return self.n_chips / self.bw

    @property
    def symbol_samples(self):
        """Number of IQ samples per symbol."""
        return int(self.symbol_duration * self.sample_rate)


def decimate_to_lora(samples, params, target_rate=None):
    """Filter and decimate IQ to near-LoRa bandwidth for cleaner de-chirping.

    Uses FFT-based brick-wall lowpass filter and downsampling.

    Args:
        samples: Complex IQ at params.sample_rate.
        params: LoraParams (original sample rate).
        target_rate: Target sample rate (default: 2 * bw).

    Returns:
        Tuple of (decimated_samples, decimated_params).
    """
    if target_rate is None:
        target_rate = 2 * params.bw

    factor = int(params.sample_rate / target_rate)
    if factor <= 1:
        return samples, params

    # Actual output rate may differ from target_rate due to integer division
    actual_rate = params.sample_rate / factor

    n = len(samples)
    n_out = n // factor
    half = n_out // 2

    # FFT-based decimation: keep only bins within ±target_rate/2
    spectrum = np.fft.fft(samples)
    truncated = np.empty(n_out, dtype=spectrum.dtype)
    truncated[:half] = spectrum[:half]       # positive frequencies
    truncated[half:] = spectrum[n - half:]   # negative frequencies

    # IFFT at reduced length; scale by n_out/n to preserve amplitude
    decimated = np.fft.ifft(truncated) * (n_out / n)

    new_params = LoraParams(sf=params.sf, bw=params.bw, sample_rate=actual_rate)
    return decimated.astype(np.complex64), new_params


def generate_chirp(params, direction='up'):
    """Generate a reference chirp signal.

    Args:
        params: LoraParams with sf, bw, sample_rate.
        direction: 'up' for up-chirp, 'down' for down-chirp.

    Returns:
        Complex64 numpy array of one symbol duration.
    """
    n_samples = params.symbol_samples
    t = np.arange(n_samples, dtype=np.float64) / params.sample_rate

    T = params.symbol_duration
    f0 = -params.bw / 2  # start frequency
    chirp_rate = params.bw / T  # Hz per second

    if direction == 'down':
        f0 = params.bw / 2
        chirp_rate = -chirp_rate

    # Instantaneous phase: integral of frequency
    # freq(t) = f0 + chirp_rate * t
    # phase(t) = 2*pi * (f0*t + chirp_rate/2 * t^2)
    phase = 2 * np.pi * (f0 * t + chirp_rate / 2 * t ** 2)
    return np.exp(1j * phase).astype(np.complex64)


def dechirp(samples, ref_chirp):
    """De-chirp by multiplying samples with conjugate of reference chirp.

    Args:
        samples: Complex IQ samples (at least len(ref_chirp) long).
        ref_chirp: Reference chirp from generate_chirp().

    Returns:
        De-chirped complex samples (same length as ref_chirp).
    """
    n = len(ref_chirp)
    return samples[:n] * np.conj(ref_chirp)


def dechirp_and_fft(samples, ref_chirp, n_fft=None):
    """De-chirp samples and compute FFT to extract symbol value.

    Args:
        samples: Complex IQ samples.
        ref_chirp: Reference chirp.
        n_fft: FFT size (defaults to len(ref_chirp)).

    Returns:
        Tuple of (fft_magnitudes, peak_bin, peak_snr_db).
        peak_bin is the symbol value (0 to N-1).
        peak_snr_db is the peak power relative to the median bin.
    """
    if n_fft is None:
        n_fft = len(ref_chirp)

    dc = dechirp(samples, ref_chirp)
    spectrum = np.fft.fft(dc, n=n_fft)
    magnitudes = np.abs(spectrum)

    peak_bin = np.argmax(magnitudes)
    peak_power = magnitudes[peak_bin]

    # SNR: peak vs median, corrected for expected noise baseline.
    # For N Rayleigh-distributed FFT bins, max/median ~ sqrt(log2(N)),
    # giving a noise baseline of 10*log10(log2(N)) dB (~10-12 dB).
    # Subtracting this gives ~0 dB on pure noise regardless of FFT size.
    median_power = np.median(magnitudes)
    if median_power > 0:
        raw_snr = 20 * np.log10(peak_power / median_power)
        n_bins = len(magnitudes)
        noise_baseline = 10 * np.log10(np.log2(n_bins)) if n_bins > 1 else 0
        snr_db = raw_snr - noise_baseline
    else:
        snr_db = 0.0

    return magnitudes, int(peak_bin), float(snr_db)


def detect_preamble(samples, params, min_chirps=4, snr_threshold=5.0,
                    n_offsets=2):
    """Scan samples for a LoRa preamble (repeated up-chirps).

    Uses overlapping windows to handle arbitrary chirp alignment.
    Looks for consecutive symbol-spaced windows with high de-chirp SNR
    (indicating chirp energy is present, regardless of exact peak bin).

    Args:
        samples: Complex IQ samples to scan.
        params: LoraParams.
        min_chirps: Minimum consecutive high-SNR windows to count as preamble.
        snr_threshold: Minimum SNR in dB for a window to count as a chirp.
        n_offsets: Number of starting offsets to try (1=fast, 2=default).

    Returns:
        List of detection dicts, each with:
          - 'sample_offset': sample index where preamble starts
          - 'n_chirps': number of consecutive chirps detected
          - 'snr_db': average SNR across detected chirps
          - 'peak_bin': median bin across detected chirps
    """
    ref_up = generate_chirp(params, direction='up')
    sym_len = params.symbol_samples
    step = sym_len // n_offsets
    n_total = len(samples)

    if n_total < sym_len * min_chirps:
        return []

    # Pre-conjugate reference for de-chirping
    ref_conj = np.conj(ref_up)

    best_detection = None
    noise_baseline = 10 * np.log10(np.log2(sym_len)) if sym_len > 1 else 0

    for start in range(0, sym_len, step):
        # Batch all windows into a 2D array for vectorized FFT
        n_windows = (n_total - start) // sym_len
        if n_windows < min_chirps:
            continue

        # Reshape signal into non-overlapping windows from this offset
        end = start + n_windows * sym_len
        windowed = samples[start:end].reshape(n_windows, sym_len)

        # Batch dechirp + FFT (one call for all windows)
        dechirped = windowed * ref_conj[np.newaxis, :]
        spectra = np.abs(np.fft.fft(dechirped, axis=1))

        # Vectorized peak finding and SNR
        peak_bins = np.argmax(spectra, axis=1)
        peak_powers = spectra[np.arange(n_windows), peak_bins]
        median_powers = np.median(spectra, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            raw_snrs = np.where(
                median_powers > 0,
                20 * np.log10(peak_powers / median_powers) - noise_baseline,
                0.0,
            )

        offsets = start + np.arange(n_windows) * sym_len
        windows = [
            {'bin': int(peak_bins[i]), 'snr': float(raw_snrs[i]), 'offset': int(offsets[i])}
            for i in range(n_windows)
        ]

        # Find runs of consecutive high-SNR windows
        detections = _find_snr_runs(windows, min_chirps, snr_threshold)

        for d in detections:
            if best_detection is None or d['snr_db'] > best_detection['snr_db']:
                best_detection = d

    return [best_detection] if best_detection else []


def _find_snr_runs(windows, min_chirps, snr_threshold):
    """Find runs of consecutive windows above SNR threshold."""
    detections = []
    run_start = None
    run_snrs = []
    run_bins = []

    for i, w in enumerate(windows):
        if w['snr'] >= snr_threshold:
            if run_start is None:
                run_start = i
                run_snrs = []
                run_bins = []
            run_snrs.append(w['snr'])
            run_bins.append(w['bin'])
        else:
            if run_start is not None and len(run_snrs) >= min_chirps:
                detections.append({
                    'sample_offset': windows[run_start]['offset'],
                    'n_chirps': len(run_snrs),
                    'snr_db': float(np.mean(run_snrs)),
                    'peak_bin': int(np.median(run_bins)),
                })
            run_start = None
            run_snrs = []
            run_bins = []

    # Check final run
    if run_start is not None and len(run_snrs) >= min_chirps:
        detections.append({
            'sample_offset': windows[run_start]['offset'],
            'n_chirps': len(run_snrs),
            'snr_db': float(np.mean(run_snrs)),
            'peak_bin': int(np.median(run_bins)),
        })

    return detections


def detect_preamble_binmatch(samples, params, min_preamble=6,
                             snr_threshold=10.0, max_bin_drift_std=50,
                             n_offsets=4):
    """Detect LoRa preamble using dechirp+FFT with drift-tolerant bin matching.

    Robust to TX/RX clock mismatch (bandwidth offset up to ~1%).

    Approach:
      1. Dechirp+FFT every symbol-length window at multiple starting offsets
      2. Find runs of consecutive high-SNR windows (signal present)
      3. Within each run, find the preamble portion: consecutive bins that
         follow a LINEAR trend (constant drift from clock mismatch).
         Data symbols have random bins, so they break the linear trend.
      4. Deduplicate overlapping detections across offsets (keep best SNR).

    The clock mismatch between TX and RX causes the dechirp bin to drift
    linearly across preamble symbols. For preamble, consecutive bin diffs
    are nearly constant (e.g., -25 +/- 8). For data, bin diffs are random
    across the full range (0 to 2^SF). The std of bin diffs cleanly
    separates preamble from data.

    Args:
        samples: Complex IQ samples to scan.
        params: LoraParams.
        min_preamble: Minimum preamble symbols to trigger detection.
        snr_threshold: Minimum dechirp SNR (dB) for signal-present.
        max_bin_drift_std: Maximum std of consecutive bin diffs to count
            as preamble (accounts for clock mismatch). Default 50 is
            generous enough for +/-1% BW offset.
        n_offsets: Number of starting offsets to try (1=fast, 4=default).
            Each offset is sym_len/n_offsets apart, ensuring at least one
            offset aligns well with any packet's chirp boundaries.

    Returns:
        List of detection dicts, each with:
          - 'sample_offset': sample index where preamble starts
          - 'n_preamble': number of preamble symbols detected
          - 'n_total': total signal symbols in the burst
          - 'snr_db': average SNR across preamble
          - 'peak_bin': median bin of preamble symbols
          - 'bin_drift': bin change per symbol (proportional to BW mismatch)
    """
    ref_up = generate_chirp(params, direction='up')
    sym_len = params.symbol_samples
    n_total = len(samples)

    if n_total < sym_len * min_preamble:
        return []

    ref_conj = np.conj(ref_up)
    noise_baseline = 10 * np.log10(np.log2(sym_len)) if sym_len > 1 else 0
    step = sym_len // n_offsets

    all_detections = []

    for start in range(0, sym_len, step):
        n_windows = (n_total - start) // sym_len
        if n_windows < min_preamble:
            continue

        end = start + n_windows * sym_len
        windowed = samples[start:end].reshape(n_windows, sym_len)
        dechirped = windowed * ref_conj[np.newaxis, :]
        spectra = np.abs(np.fft.fft(dechirped, axis=1))

        peak_bins = np.argmax(spectra, axis=1)
        peak_powers = spectra[np.arange(n_windows), peak_bins]
        median_powers = np.median(spectra, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            snrs = np.where(
                median_powers > 0,
                20 * np.log10(peak_powers / median_powers) - noise_baseline,
                0.0,
            )

        # Find runs of high-SNR windows and extract preamble from each
        run_start = None
        for i in range(n_windows + 1):
            in_signal = i < n_windows and snrs[i] >= snr_threshold
            if in_signal and run_start is None:
                run_start = i
            elif not in_signal and run_start is not None:
                run_end = i
                run_len = run_end - run_start

                if run_len >= min_preamble:
                    d = _extract_preamble(
                        peak_bins[run_start:run_end],
                        snrs[run_start:run_end],
                        start + run_start * sym_len,
                        sym_len, run_len, min_preamble, max_bin_drift_std,
                    )
                    if d is not None:
                        all_detections.append(d)

                run_start = None

    # Deduplicate: detections within 1 symbol duration of each other
    # are the same packet seen at different offsets — keep highest SNR.
    return _deduplicate_detections(all_detections, sym_len)


def _extract_preamble(run_bins, run_snrs, sample_offset, sym_len,
                      run_len, min_preamble, max_bin_drift_std):
    """Find the preamble portion within a high-SNR run using bin-diff consistency.

    Returns a detection dict or None.
    """
    bins_f = run_bins.astype(float)
    n_fft = sym_len

    bin_diffs = np.diff(bins_f)
    # Normalize diffs to [-N/2, N/2] for modular distance
    bin_diffs = (bin_diffs + n_fft / 2) % n_fft - n_fft / 2

    # Find the longest prefix where bin diffs are consistent
    best_preamble_len = 0
    best_drift = 0.0

    for end_idx in range(min_preamble, run_len + 1):
        diffs = bin_diffs[:end_idx - 1]
        if len(diffs) < 2:
            if end_idx >= min_preamble:
                best_preamble_len = end_idx
                best_drift = diffs[0] if len(diffs) > 0 else 0
            continue
        std = np.std(diffs)
        if std <= max_bin_drift_std:
            best_preamble_len = end_idx
            best_drift = np.mean(diffs)
        else:
            break

    if best_preamble_len >= min_preamble:
        preamble_snrs = run_snrs[:best_preamble_len]
        preamble_bins = bins_f[:best_preamble_len]
        return {
            'sample_offset': int(sample_offset),
            'n_preamble': best_preamble_len,
            'n_total': run_len,
            'snr_db': float(np.mean(preamble_snrs)),
            'peak_bin': int(np.median(preamble_bins)),
            'bin_drift': float(best_drift),
        }
    return None


def _deduplicate_detections(detections, sym_len):
    """Remove duplicate detections of the same packet across offsets.

    Detections within 1 symbol duration are considered the same packet.
    Keeps the one with highest SNR (best window alignment).
    """
    if not detections:
        return []

    # Sort by sample_offset
    detections.sort(key=lambda d: d['sample_offset'])

    merged = [detections[0]]
    for d in detections[1:]:
        if d['sample_offset'] - merged[-1]['sample_offset'] < sym_len:
            # Same packet — keep higher SNR
            if d['snr_db'] > merged[-1]['snr_db']:
                merged[-1] = d
        else:
            merged.append(d)

    return merged


def generate_lora_preamble(params, n_chirps=8, payload_symbols=None):
    """Generate a synthetic LoRa preamble (for testing).

    Args:
        params: LoraParams.
        n_chirps: Number of preamble chirps.
        payload_symbols: Optional list of symbol values to append after preamble.

    Returns:
        Complex64 numpy array of the full signal.
    """
    up = generate_chirp(params, direction='up')
    signal = np.tile(up, n_chirps)

    if payload_symbols:
        sym_len = params.symbol_samples
        for sym_val in payload_symbols:
            # Cyclic-shift the up-chirp by sym_val positions
            shift = int(sym_val * sym_len / params.n_chips)
            shifted = np.roll(up, -shift)
            signal = np.concatenate([signal, shifted])

    return signal
