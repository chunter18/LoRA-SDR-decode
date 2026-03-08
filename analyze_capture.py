"""Offline analysis of saved IQ captures — try different sample rate assumptions.

Usage:
  python analyze_capture.py captures/iq_0001_SF12_250k.npy
  python analyze_capture.py captures/iq_0001_SF12_250k.npy --rates 1e6,2e6
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chirp_detect import estimate_parameters


def analyze(samples, sample_rate, bw=125e3, label=""):
    """Run estimate_parameters with verbose output."""
    print(f"\n{'='*60}")
    print(f"Sample rate assumption: {sample_rate/1e6:.3f} MHz  {label}")
    print(f"Block: {len(samples)} samples = {len(samples)/sample_rate*1e3:.1f} ms")
    print(f"{'='*60}")

    result = estimate_parameters(
        samples, sample_rate,
        bw=bw,
        min_chirps=6,
        snr_threshold=5.0,
        verbose=True,
    )

    if result:
        print(f"\nWINNER: SF{result['sf']}/{result['bw']/1e3:.0f}k, "
              f"SNR={result['snr_db']:.1f}, score={result['score']:.1f}, "
              f"chirps={result['n_chirps']}, bin_std={result.get('bin_std',0):.1f}")
    else:
        print("\nNo detection.")
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze saved IQ capture')
    parser.add_argument('file', help='.npy file from --iq-save')
    parser.add_argument('--rates', default='0.5e6,1e6,1.5e6,2e6',
                        help='Comma-separated sample rates to try (default: 0.5e6,1e6,1.5e6,2e6)')
    parser.add_argument('--bw', type=float, default=125e3,
                        help='LoRa bandwidth in Hz (default: 125000)')
    args = parser.parse_args()

    samples = np.load(args.file)
    print(f"Loaded {args.file}: {len(samples)} samples, dtype={samples.dtype}")

    rates = [float(r) for r in args.rates.split(',')]
    results = {}
    for rate in rates:
        r = analyze(samples, rate, bw=args.bw)
        if r:
            results[rate] = r

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY — Best detection at each sample rate:")
        print(f"{'='*60}")
        for rate, r in sorted(results.items()):
            print(f"  {rate/1e6:.3f} MSPS -> SF{r['sf']}/{r['bw']/1e3:.0f}k, "
                  f"score={r['score']:.1f}, chirps={r['n_chirps']}, "
                  f"bin_std={r.get('bin_std',0):.1f}")


if __name__ == '__main__':
    main()
