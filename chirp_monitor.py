"""
LoRa Chirp Monitor — CLI tool that detects LoRa preambles from live SDR input.

Streams IQ from the Pluto SDR, runs chirp detection, and prints detections
to stdout. No GUI.

Usage:
  python chirp_monitor.py
  python chirp_monitor.py --sf 7            # detect SF7 only
  python chirp_monitor.py --backend rtl_433  # use rtl_433 instead of rx_sdr
"""

import sys
import argparse
import time
import numpy as np

from sdr_source import start_sdr, read_iq_blocks, DEFAULT_FREQ, DEFAULT_SAMPLE_RATE, DEFAULT_BANDWIDTH
from chirp_detect import LoraParams, detect_preamble_binmatch, generate_chirp, dechirp_and_fft

try:
    from chirp_demod import demodulate
except ImportError:
    demodulate = None


class Deduplicator:
    """Suppress duplicate detections of the same packet across overlapping windows.

    Uses absolute sample position in the IQ stream (not wall-clock time)
    so deduplication is immune to SDR buffering and processing delays.
    """

    def __init__(self, min_gap_samples):
        self._last_sample = {}  # sf -> absolute sample offset of last report
        self.min_gap = min_gap_samples

    def should_report(self, sf, abs_sample_offset):
        last = self._last_sample.get(sf)
        if last is not None and abs(abs_sample_offset - last) < self.min_gap:
            return False
        self._last_sample[sf] = abs_sample_offset
        return True


def estimate_noise_floor_dbfs(samples):
    """Estimate noise floor as mean power in dBFS.

    Args:
        samples: Complex IQ samples.

    Returns:
        Noise floor in dBFS (float).
    """
    mean_power = np.mean(np.abs(samples) ** 2)
    if mean_power > 0:
        return float(10 * np.log10(mean_power))
    return -np.inf


def main():
    parser = argparse.ArgumentParser(description='LoRa chirp detector')
    parser.add_argument('--freq', default=DEFAULT_FREQ, help='Center frequency (default: 915M)')
    parser.add_argument('--sf', default='7-12', help='Spreading factor or range (default: 7-12)')
    parser.add_argument('--bw', type=float, default=125e3, help='LoRa bandwidth in Hz (default: 125000)')
    parser.add_argument('--threshold', type=float, default=5.0, help='SNR threshold in dB (default: 5.0)')
    parser.add_argument('--demod', action='store_true', help='Extract raw symbols after preamble detection')
    parser.add_argument('--debug', action='store_true', help='Print per-block peak SNR for diagnostics')
    parser.add_argument('--backend', default='rx_sdr', choices=['rx_sdr', 'rtl_433'],
                        help='SDR backend (default: rx_sdr)')
    args = parser.parse_args()

    # Parse SF range
    if '-' in args.sf:
        sf_min, sf_max = map(int, args.sf.split('-'))
        sf_list = list(range(sf_min, sf_max + 1))
    else:
        sf_list = [int(args.sf)]

    sample_rate = DEFAULT_SAMPLE_RATE

    # Build params for each SF
    params_list = [
        LoraParams(sf=sf, bw=args.bw, sample_rate=sample_rate)
        for sf in sf_list
    ]

    # Use the largest symbol size for block reading
    max_sym_samples = max(p.symbol_samples for p in params_list)

    # Sliding window: analyze window_size samples, advance by step_size.
    # Window must hold a full packet (preamble + data ≈ 26 symbols).
    # Step is how much new data we read each iteration.
    window_syms = 35 if args.demod else 26
    step_syms = 8  # advance 8 symbols (~262ms at SF12) per iteration
    window_size = max_sym_samples * window_syms
    step_size = max_sym_samples * step_syms

    print(f"LoRa Chirp Monitor")
    print(f"  Backend: {args.backend}")
    print(f"  Frequency: {args.freq}")
    print(f"  Spreading factors: {sf_list}")
    print(f"  Bandwidth: {args.bw/1e3:.0f} kHz")
    print(f"  SNR threshold: {args.threshold:.1f} dB")
    print(f"  Window: {window_size} samples ({window_size/sample_rate*1e3:.0f} ms), "
          f"step: {step_size} samples ({step_size/sample_rate*1e3:.0f} ms)")
    print()

    print("Launching SDR...")
    try:
        proc = start_sdr(args.freq, sample_rate, DEFAULT_BANDWIDTH, backend=args.backend)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("Listening for LoRa chirps... (Ctrl+C to stop)")
    print()

    detection_count = 0
    sf_counts = {}  # sf -> count
    block_count = 0
    start_time = time.time()
    # Deduplicate by sample position; min gap = 1 full packet (~26 symbols)
    dedup = Deduplicator(min_gap_samples=max_sym_samples * 26)

    # Pre-allocated sliding window buffer
    ring = np.empty(window_size, dtype=np.complex64)
    ring_fill = 0  # how many valid samples in ring
    total_samples_read = 0

    try:
        for chunk in read_iq_blocks(proc, block_size=step_size):
            total_samples_read += len(chunk)
            n = len(chunk)

            if ring_fill < window_size:
                # Still filling initial window
                end = min(ring_fill + n, window_size)
                ring[ring_fill:end] = chunk[:end - ring_fill]
                ring_fill = end
                if ring_fill < window_size:
                    continue
            else:
                # Shift old data left, append new chunk at the end
                ring[:window_size - n] = ring[n:]
                ring[window_size - n:] = chunk

            samples = ring
            block_count += 1
            elapsed = time.time() - start_time
            window_start_sample = total_samples_read - window_size

            # Noise floor estimate
            noise_dbfs = estimate_noise_floor_dbfs(samples)
            if args.debug:
                print(f"[{elapsed:8.1f}s] noise floor: {noise_dbfs:.1f} dBFS", file=sys.stderr)

            for params in params_list:
                # Debug: show per-window de-chirp results
                if args.debug:
                    ref_conj = np.conj(generate_chirp(params, direction='up'))
                    sym_len = params.symbol_samples
                    step = sym_len // 4
                    for start in range(0, sym_len, step):
                        wins = []
                        off = start
                        while off + sym_len <= len(samples):
                            w = samples[off:off + sym_len]
                            spec = np.abs(np.fft.fft(w * ref_conj))
                            pk = int(np.argmax(spec))
                            med = np.median(spec)
                            if med > 0:
                                raw = 20 * np.log10(spec[pk] / med)
                                nb = 10 * np.log10(np.log2(len(spec)))
                                snr = float(raw - nb)
                            else:
                                snr = 0
                            wins.append(f"{snr:5.1f}@{pk}")
                            off += sym_len
                        print(
                            f"[{elapsed:8.1f}s] SF{params.sf} offset={start:6d}: "
                            + " | ".join(wins),
                            file=sys.stderr,
                        )

                if args.demod and demodulate is not None:
                    results = demodulate(
                        samples, params,
                        n_data_symbols=20,
                        min_chirps=6,
                        snr_threshold=args.threshold,
                    )
                    for r in results:
                        abs_offset = window_start_sample + r.get('sample_offset', 0)
                        if not dedup.should_report(params.sf, abs_offset):
                            continue
                        detection_count += 1
                        sf_counts[params.sf] = sf_counts.get(params.sf, 0) + 1
                        syms = r['symbols']
                        print(
                            f"[{elapsed:8.1f}s] "
                            f"SF={params.sf}, "
                            f"chirps={r['n_chirps']}, "
                            f"SNR={r['snr_db']:.1f} dB, "
                            f"noise={noise_dbfs:.1f} dBFS, "
                            f"{len(syms)} symbols: {syms}"
                        )
                else:
                    detections = detect_preamble_binmatch(
                        samples, params,
                        min_preamble=6,
                        snr_threshold=args.threshold,
                        n_offsets=2,
                    )

                    for d in detections:
                        abs_offset = window_start_sample + d['sample_offset']
                        if not dedup.should_report(params.sf, abs_offset):
                            continue
                        detection_count += 1
                        sf_counts[params.sf] = sf_counts.get(params.sf, 0) + 1

                        # Convert peak bin to frequency offset
                        n_fft = params.symbol_samples
                        peak_bin = d['peak_bin']
                        if peak_bin >= n_fft // 2:
                            offset_hz = (peak_bin - n_fft) * sample_rate / n_fft
                        else:
                            offset_hz = peak_bin * sample_rate / n_fft
                        center_hz = float(str(args.freq).replace('M', 'e6').replace('k', 'e3'))
                        freq_mhz = (center_hz + offset_hz) / 1e6

                        data_time = abs_offset / sample_rate
                        print(
                            f"[{data_time:8.1f}s] "
                            f"LoRa preamble: "
                            f"SF={params.sf}, "
                            f"freq={freq_mhz:.4f} MHz, "
                            f"preamble={d['n_preamble']}, "
                            f"total={d['n_total']} syms, "
                            f"SNR={d['snr_db']:.1f} dB, "
                            f"drift={d['bin_drift']:.1f}, "
                            f"noise={noise_dbfs:.1f} dBFS"
                        )

            # Periodic status
            if block_count % 50 == 0:
                sf_summary = " ".join(f"SF{sf}:{n}" for sf, n in sorted(sf_counts.items()))
                print(
                    f"[{elapsed:8.1f}s] "
                    f"... {detection_count} detections in {block_count} blocks"
                    f" [{sf_summary}]",
                    file=sys.stderr,
                )

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\nStopped after {elapsed:.1f}s. {detection_count} detections in {block_count} blocks.")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == '__main__':
    main()
