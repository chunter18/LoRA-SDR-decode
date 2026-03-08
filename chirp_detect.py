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

    new_params = LoraParams(sf=params.sf, bw=params.bw, sample_rate=target_rate)
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


def detect_preamble(samples, params, min_chirps=4, snr_threshold=5.0):
    """Scan samples for a LoRa preamble (repeated up-chirps).

    Uses overlapping windows to handle arbitrary chirp alignment.
    Looks for consecutive symbol-spaced windows with high de-chirp SNR
    (indicating chirp energy is present, regardless of exact peak bin).

    Args:
        samples: Complex IQ samples to scan.
        params: LoraParams.
        min_chirps: Minimum consecutive high-SNR windows to count as preamble.
        snr_threshold: Minimum SNR in dB for a window to count as a chirp.

    Returns:
        List of detection dicts, each with:
          - 'sample_offset': sample index where preamble starts
          - 'n_chirps': number of consecutive chirps detected
          - 'snr_db': average SNR across detected chirps
          - 'peak_bin': median bin across detected chirps
    """
    ref_up = generate_chirp(params, direction='up')
    sym_len = params.symbol_samples
    step = sym_len // 2  # 50% overlap for alignment tolerance
    n_total = len(samples)

    if n_total < sym_len * min_chirps:
        return []

    # Pre-conjugate reference for de-chirping
    ref_conj = np.conj(ref_up)

    # Try multiple starting offsets (quarter-symbol steps)
    best_detection = None
    best_score = -np.inf

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
            # Prefer detections with consistent bins (low bin_std)
            d_score = d['snr_db'] / (1 + d.get('bin_std', 0) / 5.0)
            if d_score > best_score:
                best_detection = d
                best_score = d_score

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
                # bin_std from first 8 windows only (preamble region, before
                # sync/SFD/data change the peak bin)
                preamble_bins = run_bins[:8]
                detections.append({
                    'sample_offset': windows[run_start]['offset'],
                    'n_chirps': len(run_snrs),
                    'snr_db': float(np.mean(run_snrs)),
                    'peak_bin': int(np.median(preamble_bins)),
                    'bin_std': float(np.std(preamble_bins)),
                })
            run_start = None
            run_snrs = []
            run_bins = []

    # Check final run
    if run_start is not None and len(run_snrs) >= min_chirps:
        preamble_bins = run_bins[:8]
        detections.append({
            'sample_offset': windows[run_start]['offset'],
            'n_chirps': len(run_snrs),
            'snr_db': float(np.mean(run_snrs)),
            'peak_bin': int(np.median(preamble_bins)),
            'bin_std': float(np.std(preamble_bins)),
        })

    return detections


def bin_to_freq_offset(peak_bin, n_fft, sample_rate):
    """Convert FFT peak bin to frequency offset in Hz.

    Handles wrapping: bins above n_fft/2 map to negative frequencies.
    """
    if peak_bin > n_fft // 2:
        peak_bin -= n_fft
    return peak_bin * sample_rate / n_fft


def estimate_parameters(samples, sample_rate,
                        sf_candidates=None, bw=125e3,
                        min_chirps=4, snr_threshold=5.0,
                        verbose=False):
    """Estimate LoRa SF and frequency offset by scanning SFs at a fixed BW.

    Args:
        samples: Complex IQ samples.
        sample_rate: SDR sample rate in Hz.
        sf_candidates: List of SFs to try (default: [7..12]).
        bw: LoRa bandwidth in Hz (default: 125e3). Fixed, not auto-detected.
        min_chirps: Minimum preamble chirps for detection.
        snr_threshold: SNR threshold in dB.
        verbose: If True, print all candidate scores to stderr.

    Returns:
        Dict with 'sf', 'bw', 'freq_offset_hz', 'snr_db', 'n_chirps',
        'sample_offset', 'peak_bin' — or None if nothing detected.
    """
    if sf_candidates is None:
        sf_candidates = list(range(7, 13))

    best = None
    best_score = -np.inf
    for sf in sf_candidates:
        params = LoraParams(sf=sf, bw=bw, sample_rate=sample_rate)
        if params.symbol_samples * min_chirps > len(samples):
            continue
        detections = detect_preamble(samples, params,
                                     min_chirps=min_chirps,
                                     snr_threshold=snr_threshold)
        for det in detections:
            # Score: SNR penalized by bin inconsistency and chirp count.
            # For correct SF, preamble bins are consistent (std ~0-5)
            # and chirp count is 6-12. Wrong SF/BW produces wandering bins
            # and/or very high chirp counts (many windows fit in one preamble).
            bin_std = det.get('bin_std', 0)
            n = det['n_chirps']
            chirp_factor = min(n, 12) / n  # 1.0 for n<=12, shrinks for n>12
            score = det['snr_db'] / (1 + bin_std / 5.0) * chirp_factor
            if verbose:
                import sys as _sys
                print(
                    f"  SF{sf}/{bw/1e3:.0f}k: "
                    f"SNR={det['snr_db']:.1f}, "
                    f"chirps={n}, "
                    f"bin_std={bin_std:.1f}, "
                    f"chirp_factor={chirp_factor:.2f}, "
                    f"score={score:.1f}, "
                    f"bin={det['peak_bin']}",
                    file=_sys.stderr,
                )
            if score > best_score:
                best_score = score
                freq_hz = bin_to_freq_offset(det['peak_bin'],
                                              params.symbol_samples,
                                              sample_rate)
                best = {
                    'sf': sf,
                    'bw': bw,
                    'freq_offset_hz': freq_hz,
                    'snr_db': det['snr_db'],
                    'n_chirps': det['n_chirps'],
                    'sample_offset': det['sample_offset'],
                    'peak_bin': det['peak_bin'],
                    'bin_std': bin_std,
                    'score': score,
                }
    return best


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


def generate_lora_packet(params, n_preamble=8, sync_word=0x12,
                         data_symbols=None):
    """Generate a realistic LoRa packet with preamble, sync, SFD, and data.

    Mimics the SX1276 packet structure:
      - n_preamble up-chirps (unmodulated)
      - 2 sync word chirps (shifted by sync_word nibbles)
      - 2.25 SFD down-chirps
      - data symbols (shifted up-chirps)

    Args:
        params: LoraParams.
        n_preamble: Number of preamble up-chirps (default 8).
        sync_word: LoRa sync word byte (default 0x12).
        data_symbols: List of data symbol values (default: 5 random symbols).

    Returns:
        Complex64 numpy array of the full packet.
    """
    up = generate_chirp(params, direction='up')
    down = generate_chirp(params, direction='down')
    sym_len = params.symbol_samples

    # Preamble: n unmodulated up-chirps
    signal = np.tile(up, n_preamble)

    # Sync word: 2 chirps shifted by sync_word nibbles
    # SX1276 uses high and low nibbles of sync_word
    sync_hi = (sync_word >> 4) & 0x0F
    sync_lo = sync_word & 0x0F
    for nibble in [sync_hi, sync_lo]:
        shift = int(nibble * sym_len / params.n_chips) * (params.n_chips // 16)
        shifted = np.roll(up, -shift)
        signal = np.concatenate([signal, shifted])

    # SFD: 2.25 down-chirps
    signal = np.concatenate([signal, down, down, down[:sym_len // 4]])

    # Data symbols
    if data_symbols is None:
        rng = np.random.default_rng(42)
        data_symbols = rng.integers(0, params.n_chips, size=5).tolist()
    for sym_val in data_symbols:
        shift = int(sym_val * sym_len / params.n_chips)
        shifted = np.roll(up, -shift)
        signal = np.concatenate([signal, shifted])

    return signal
