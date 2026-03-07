"""
LoRa Chirp Monitor — CLI tool that detects LoRa preambles from live SDR input.

Launches rtl_433 to stream IQ from the Pluto SDR, runs chirp detection,
and prints detections to stdout. No GUI.

Usage:
  python chirp_monitor.py
  python chirp_monitor.py --sf 7       # detect SF7 only
  python chirp_monitor.py --sf 7-12    # scan all spreading factors (default)
"""

import sys
import argparse
import time
import numpy as np

from sdr_source import start_sdr, read_iq_blocks, DEFAULT_FREQ, DEFAULT_SAMPLE_RATE, DEFAULT_BANDWIDTH
from chirp_detect import LoraParams, detect_preamble, generate_chirp, dechirp_and_fft


class Deduplicator:
    """Suppress duplicate detections within one preamble duration per SF."""

    def __init__(self):
        self._last_time = {}  # sf -> timestamp

    def should_report(self, sf, timestamp, preamble_duration):
        last = self._last_time.get(sf)
        if last is not None and (timestamp - last) < preamble_duration:
            return False
        self._last_time[sf] = timestamp
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
    parser.add_argument('--threshold', type=float, default=15.0, help='SNR threshold in dB (default: 15.0)')
    parser.add_argument('--debug', action='store_true', help='Print per-block peak SNR for diagnostics')
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
    # Read blocks large enough for at least 10 symbols of the largest SF
    block_size = max_sym_samples * 10

    print(f"LoRa Chirp Monitor")
    print(f"  Frequency: {args.freq}")
    print(f"  Spreading factors: {sf_list}")
    print(f"  Bandwidth: {args.bw/1e3:.0f} kHz")
    print(f"  SNR threshold: {args.threshold:.1f} dB")
    print(f"  Block size: {block_size} samples ({block_size/sample_rate*1e3:.1f} ms)")
    print()

    print("Launching SDR...")
    try:
        proc = start_sdr(args.freq, sample_rate, DEFAULT_BANDWIDTH)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("Listening for LoRa chirps... (Ctrl+C to stop)")
    print()

    detection_count = 0
    block_count = 0
    start_time = time.time()
    dedup = Deduplicator()

    try:
        for samples in read_iq_blocks(proc, block_size=block_size):
            block_count += 1
            elapsed = time.time() - start_time

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
                            snr = float(20 * np.log10(spec[pk] / med)) if med > 0 else 0
                            wins.append(f"{snr:5.1f}@{pk}")
                            off += sym_len
                        print(
                            f"[{elapsed:8.1f}s] SF{params.sf} offset={start:6d}: "
                            + " | ".join(wins),
                            file=sys.stderr,
                        )

                detections = detect_preamble(
                    samples, params,
                    min_chirps=6,
                    snr_threshold=args.threshold,
                )

                for d in detections:
                    # Suppress window: preamble + block duration (a preamble
                    # spanning two blocks produces two detections)
                    suppress_dur = 8 * params.symbol_duration + block_size / sample_rate
                    if not dedup.should_report(params.sf, elapsed, suppress_dur):
                        continue
                    detection_count += 1
                    print(
                        f"[{elapsed:8.1f}s] "
                        f"LoRa preamble detected: "
                        f"SF={params.sf}, "
                        f"chirps={d['n_chirps']}, "
                        f"SNR={d['snr_db']:.1f} dB, "
                        f"bin={d['peak_bin']}, "
                        f"noise={noise_dbfs:.1f} dBFS"
                    )

            # Periodic status
            if block_count % 50 == 0:
                print(
                    f"[{elapsed:8.1f}s] "
                    f"... {detection_count} detections in {block_count} blocks",
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
