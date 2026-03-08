# LoRa Chirp Detector — Backlog

## Current Status
- Preamble detection WORKS at real-time (1.0 MSPS) via rx_sdr backend
- Multi-SF scanning (SF7-12) runs at real-time, but has cross-SF false positives
- Symbol demodulation WORKS on synthetic signals (all SFs, with noise)
- Beacon: RFM95W on Arduino, SF7/125kHz or SF7-12 cycle, "HELLO" every 3 seconds
- SDR: Pluto via rx_sdr (CS16 format, 1 MHz sample rate)
- 55/55 tests passing

## SDR Backend
Switched from rtl_433 to rx_sdr as default IQ source. rtl_433 runs its full
signal processing pipeline (AM/FM demod, pulse detection, filtering) even when
only dumping raw IQ with `-w -`, limiting throughput to 0.6 MSPS. rx_sdr is a
lightweight SoapySDR streamer that achieves full 1.0 MSPS. See
`sdr_throughput_findings.md` for details. Old backend available via
`--backend rtl_433`.

## SF Scan Mode (Provisional)
Multi-SF detection (SF7-12) runs at real-time by:
- Block size: 10 × max_symbol_samples (327,680 for SF12 = 328 ms)
- Single alignment offset when scanning (vs 2 for single-SF)
- Batch FFT per SF per block

**Known accuracy issue:** Cross-SF false positives. A beacon transmitting at
SF7 can produce false detections at SF8, SF9, etc. Confirmed by turning off
the beacon — zero detections with beacon off, ruling out other devices.
There is no reliable way to validate SF accuracy without decoding the payload.
Plan: get decoding working, then transmit the SF value in the payload and
measure detection accuracy against ground truth.

## Completed

### Deduplication
Preambles spanning two blocks produced two detections ~0.7s apart.
Fixed: `Deduplicator` class suppresses within (preamble + block duration).

### SNR Metric
Corrected peak/median: `raw - 10*log10(log2(N_bins))`. Noise reads ~0 dB,
signal ~6-11 dB. Consistent across FFT sizes.

### Symbol Demodulation (synthetic only)
chirp_demod.py: `generate_lora_frame()`, `find_sfd()`, `extract_symbols()`,
`demodulate()`. Not yet tested on live signals.

### Real-time Throughput
rx_sdr backend delivers 1.0 MSPS. Processing at 10-13 MSPS (single SF) or
~3 MSPS (6 SFs) — well within budget. See `sdr_throughput_findings.md`.

## Next Steps (in order)

### Step 1: LoRa Protocol Decode
Convert raw symbols to bytes. Multiple layers:

1. **De-gray code** — LoRa gray-codes symbols to minimize bit errors
2. **De-interleave** — symbols are interleaved in blocks of (4+CR) across SF bits
3. **Hamming FEC decode** — each codeword is (4+CR) bits, CR=1 means (5,4) Hamming
4. **De-whiten** — XOR with LFSR pseudo-random sequence
5. **Header decode** — first 8 symbols use reduced rate (SF-2 bits), contain
   payload length and CR
6. **CRC check** — CRC-16 on payload

References:
- Matt Knight's talk "Decoding LoRa" (DEF CON 24)
- gr-lora by rpp0 (GNU Radio LoRa decoder, MIT license)
- https://github.com/tapparelj/gr-lora_sdr

### Step 2: Decode "HELLO"
Full pipeline from raw IQ to decoded payload bytes. Once working, modify
beacon to transmit SF value in payload and use that as ground truth to
measure SF detection accuracy.

### Step 3: Validate SF Detection
With decode working, transmit known SF in payload, compare detected SF vs
decoded SF. Use this to tune thresholds and fix cross-SF false positives.

## Architecture Notes

### File layout
```
lora/
  sdr_source.py        — IQ from rx_sdr or rtl_433 (done)
  chirp_detect.py      — preamble detection (done)
  chirp_demod.py       — symbol extraction (done, synthetic only)
  lora_decode.py       — NEW: gray/interleave/hamming/whiten/CRC
  chirp_waterfall.py   — visualization (done)
  chirp_monitor.py     — CLI detector with SF scan (done)
  chirp_decode.py      — NEW: full pipeline CLI
  sdr_throughput_findings.md — rtl_433 vs rx_sdr analysis
  arduino/
    lora_beacon_simple/    — fixed SF12/125k beacon
    lora_beacon_SF_cycle/  — cycles SF7-12
  tests/
    test_chirp_detect.py   (done, 27 tests)
    test_sdr_source.py     (done, 7 tests)
    test_chirp_demod.py    (done, 12 tests)
    test_chirp_monitor.py  (done, 9 tests)
    test_lora_decode.py    — NEW
```

### Key parameters confirmed from live testing
- rx_sdr backend: full 1.0 MSPS throughput
- rtl_433 backend: 0.6 MSPS (internal DSP overhead)
- Noise floor SNR: ~0 dB (corrected metric)
- Real chirp SNR: ~6-11 dB
- Working threshold: 4-5 dB
- Frequency offset: ~30-50 kHz between Pluto and RFM95W crystals
- Beacon interval: 3 seconds
- Block size: 327,680 samples (328 ms) for multi-SF scan
- Single alignment offset sufficient for 8-chirp preambles
