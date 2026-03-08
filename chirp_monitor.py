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
import os
import argparse
import time
import numpy as np

from sdr_source import start_sdr, read_iq_blocks, DEFAULT_FREQ, DEFAULT_SAMPLE_RATE, DEFAULT_BANDWIDTH
from chirp_detect import LoraParams, detect_preamble, generate_chirp, dechirp_and_fft, estimate_parameters
from chirp_demod import demodulate


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
    parser.add_argument('--threshold', type=float, default=5.0, help='SNR threshold in dB (default: 5.0)')
    parser.add_argument('--demod', action='store_true', help='Extract raw symbols after preamble detection')
    parser.add_argument('--scan', action='store_true', help='Auto-detect SF and frequency (scans SF7-12 at fixed BW)')
    parser.add_argument('--debug', action='store_true', help='Print per-block peak SNR for diagnostics')
    parser.add_argument('--iq-save', type=str, default=None, help='Save IQ blocks on detection to this directory')
    parser.add_argument('--sample-rate', type=float, default=None, help='Override assumed sample rate (Hz)')
    args = parser.parse_args()

    # Parse SF range
    if '-' in args.sf:
        sf_min, sf_max = map(int, args.sf.split('-'))
        sf_list = list(range(sf_min, sf_max + 1))
    else:
        sf_list = [int(args.sf)]

    sample_rate = args.sample_rate if args.sample_rate else DEFAULT_SAMPLE_RATE

    # Build params for each SF
    params_list = [
        LoraParams(sf=sf, bw=args.bw, sample_rate=sample_rate)
        for sf in sf_list
    ]

    # Use the largest symbol size for block reading
    if args.scan:
        # In scan mode, need blocks large enough for the slowest SF at the configured BW.
        # 10 symbols of SF12 gives enough room for 8-chirp preamble + alignment margin
        # while keeping blocks small enough for near-real-time processing.
        max_sym_samples = int((1 << 12) / args.bw * sample_rate)  # SF12 at configured BW
        n_symbols_per_block = 10
    else:
        max_sym_samples = max(p.symbol_samples for p in params_list)
        # With --demod, need room for preamble (8) + sync (2) + SFD (2.25) + data (~20)
        n_symbols_per_block = 35 if args.demod else 10
    block_size = max_sym_samples * n_symbols_per_block

    # Parse SDR center frequency for signal freq calculation
    freq_str = args.freq
    if freq_str.upper().endswith('M'):
        sdr_center_hz = float(freq_str[:-1]) * 1e6
    elif freq_str.upper().endswith('K'):
        sdr_center_hz = float(freq_str[:-1]) * 1e3
    else:
        sdr_center_hz = float(freq_str)

    print(f"LoRa Chirp Monitor")
    print(f"  Frequency: {args.freq}")
    if args.scan:
        print(f"  Mode: SF SCAN (SF7-12 at {args.bw/1e3:.0f} kHz BW)")
    else:
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
    bytes_received = 0
    start_time = time.time()
    dedup = Deduplicator()

    try:
        for samples in read_iq_blocks(proc, block_size=block_size):
            block_count += 1
            bytes_received += len(samples) * 4  # CS16: 4 bytes per complex sample
            elapsed = time.time() - start_time

            # Noise floor estimate
            noise_dbfs = estimate_noise_floor_dbfs(samples)
            if args.debug:
                print(f"[{elapsed:8.1f}s] noise floor: {noise_dbfs:.1f} dBFS", file=sys.stderr)

            if args.scan:
                # Auto-detect SF and frequency at fixed BW
                result = estimate_parameters(
                    samples, sample_rate,
                    bw=args.bw,
                    min_chirps=6,
                    snr_threshold=args.threshold,
                    verbose=args.debug,
                )
                if result is not None and result.get('score', 0) >= 3.0:
                    sig_freq = sdr_center_hz + result['freq_offset_hz']
                    # Dedup key: use detected SF
                    det_params = LoraParams(sf=result['sf'], bw=result['bw'],
                                           sample_rate=sample_rate)
                    suppress_dur = 8 * det_params.symbol_duration + block_size / sample_rate
                    if dedup.should_report(result['sf'], elapsed, suppress_dur):
                        detection_count += 1
                        print(
                            f"[{elapsed:8.1f}s] "
                            f"LoRa detected: "
                            f"SF={result['sf']}, "
                            f"BW={result['bw']/1e3:.0f} kHz, "
                            f"freq={sig_freq/1e6:.4f} MHz, "
                            f"chirps={result['n_chirps']}, "
                            f"SNR={result['snr_db']:.1f} dB, "
                            f"bin_std={result.get('bin_std', 0):.1f}, "
                            f"score={result.get('score', 0):.1f}"
                        )
                        if args.iq_save:
                            os.makedirs(args.iq_save, exist_ok=True)
                            fname = os.path.join(
                                args.iq_save,
                                f"iq_{detection_count:04d}_SF{result['sf']}_{result['bw']/1e3:.0f}k.npy"
                            )
                            np.save(fname, samples)
                            print(f"  -> saved {fname}", file=sys.stderr)
            else:
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

                if args.demod:
                    results = demodulate(
                        samples, params,
                        n_data_symbols=20,
                        min_chirps=6,
                        snr_threshold=args.threshold,
                    )
                    if not results and args.debug:
                        # Check if raw detection works (without decimation)
                        raw_det = detect_preamble(samples, params, min_chirps=6, snr_threshold=args.threshold)
                        if raw_det:
                            print(f"[{elapsed:8.1f}s] DEBUG: raw detection OK (SNR={raw_det[0]['snr_db']:.1f}) but demodulate() returned empty", file=sys.stderr)
                    for r in results:
                        suppress_dur = 8 * params.symbol_duration + block_size / sample_rate
                        if not dedup.should_report(params.sf, elapsed, suppress_dur):
                            continue
                        detection_count += 1
                        syms = r['symbols']
                        print(
                            f"[{elapsed:8.1f}s] "
                            f"SF={params.sf}, "
                            f"chirps={r['n_chirps']}, "
                            f"SNR={r['snr_db']:.1f} dB, "
                            f"noise={noise_dbfs:.1f} dBFS, "
                            f"offset={r.get('freq_offset', '?')}, "
                            f"data@{r['data_offset']}, "
                            f"{len(syms)} symbols: {syms}"
                        )
                else:
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
                measured_rate = bytes_received / 4 / elapsed if elapsed > 0 else 0
                rt_ratio = measured_rate / sample_rate if sample_rate > 0 else 0
                print(
                    f"[{elapsed:8.1f}s] "
                    f"... {detection_count} detections in {block_count} blocks, "
                    f"{measured_rate/1e6:.3f} MSPS "
                    f"({rt_ratio:.2f}x real-time)",
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
