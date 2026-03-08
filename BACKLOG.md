# LoRa Chirp Detector — Backlog

## Current Status
- Preamble detection WORKS reliably at real-time (1.0 MSPS) via rx_sdr backend
- Tested: 84% detection rate over 2 minutes, zero false positives (SF12, live)
- Robust to BW mismatch (~0.6% / 6400 ppm) via drift-tolerant bin matching
- SDR: Pluto via rx_sdr (CS16 format, 1 MHz sample rate, manual gain required)
- Beacon: RFM95W on Arduino, SF12/125kHz, "HELLO" every 3 seconds
- 37/37 tests passing

## Detection Architecture
- `detect_preamble_binmatch()` — primary detector, drift-tolerant
  - Multi-offset (default 2) for arbitrary packet alignment
  - Batch dechirp+FFT (vectorized numpy)
  - High-SNR run detection, then bin-diff consistency to separate preamble from data
  - Handles up to ~1% TX/RX bandwidth mismatch
- `chirp_monitor.py` — sliding window (26-sym window, 8-sym step)
  - Pre-allocated ring buffer, ~70ms processing per 262ms step
  - Sample-position deduplication (not wall-clock)

## SDR Backend
rx_sdr as default IQ source. rtl_433 runs its full signal processing pipeline
even when dumping raw IQ, limiting throughput to 0.6 MSPS. rx_sdr achieves
full 1.0 MSPS. See `sdr_throughput_findings.md`. Old backend available via
`--backend rtl_433`.

## Completed

### Drift-Tolerant Preamble Detection
Real RFM95W BW is ~124,210 Hz (not 125,000 Hz) due to SX1276 RC filter
calibration. This causes ~25 bins/symbol drift at 1 MSPS. The bin-match
detector handles this by looking for consistent (linear) bin diffs across
preamble symbols. `max_bin_drift_std=50` handles up to ~1% BW mismatch.

### Sliding Window with Sample-Based Dedup
Fixed data loss from np.concatenate overhead and wall-clock deduplication.
Pre-allocated ring buffer keeps processing at 27% of real-time. Deduplication
uses absolute sample position in the IQ stream, immune to SDR buffering jitter.

### decimate_to_lora() Sample Rate Fix
`actual_rate = sample_rate / factor` instead of `target_rate`. Fixes 53-sample
sym_len error when BW != 125000 Hz.

### AGC Problem Identified
Pluto AGC (slow_attack) destroys chirp signals. Gain must be set manually via
iio_attr before capture. Settings do NOT persist across resets.

### Autocorrelation Approach Abandoned
Schmidl-Cox autocorrelation (as used by gr-lora/rpp0) doesn't work at
oversampled rates — chirp autocorrelation peaks are only ~8 samples wide at
1 MSPS, so any timing mismatch kills correlation. Dechirp+FFT+bin matching
is the correct approach at oversampled rates.

### SNR Metric
Corrected peak/median: `raw - 10*log10(log2(N_bins))`. Noise reads ~0 dB,
signal ~7-12 dB typical, 25-30 dB close range.

### Symbol Demodulation (synthetic only)
chirp_demod.py: `generate_lora_frame()`, `find_sfd()`, `extract_symbols()`,
`demodulate()`. Not yet tested on live signals.

## Next Steps (in order)

### Step 1: Validate Demodulation on Live IQ
Test chirp_demod.py against real beacon packets. Need to handle:
- BW mismatch (dechirp bin offset)
- Fine symbol timing (currently uses coarse window alignment)
- SFD detection on real signals

### Step 2: LoRa Protocol Decode
Convert raw symbols to bytes:
1. De-gray code
2. De-interleave (blocks of 4+CR across SF bits)
3. Hamming FEC decode
4. De-whiten (XOR with LFSR)
5. Header decode (first 8 symbols, reduced rate)
6. CRC-16 check

### Step 3: Decode "HELLO"
Full pipeline from raw IQ to decoded payload bytes.

### Step 4: C Port
Translate working Python to C for rtl_433 integration.

## Key Parameters (from live testing)
- rx_sdr backend: full 1.0 MSPS throughput
- Noise floor SNR: ~0 dB (corrected metric)
- Real chirp SNR: ~25-30 dB (close range), ~7-12 dB (moderate distance)
- Working threshold: 5-10 dB
- Frequency offset: ~30-50 kHz between Pluto and RFM95W crystals
- RFM95W actual BW: ~124,210 Hz (not 125,000)
- Bin drift: ~25 bins/symbol from BW mismatch
- Beacon interval: 3 seconds (TX duration ~850ms at SF12)
- Detection processing: ~70ms per 262ms step (27% of real-time)
