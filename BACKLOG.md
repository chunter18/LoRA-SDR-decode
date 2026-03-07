# LoRa Chirp Detector — Backlog

## Current Status
- Preamble detection WORKS on live SF12 beacon (~19 dB SNR, 6-9 chirps per detection)
- Symbol demodulation WORKS on synthetic signals (all SFs, with noise)
- Beacon: RFM95W on Arduino, SF12/125kHz, "HELLO" every 3 seconds
- SDR: Pluto via rtl_433 -w - (CS16 format, 1 MHz sample rate)
- 55/55 tests passing

## Known Issues to Fix First

### ~~1. Deduplicate detections~~ DONE
Preambles spanning two blocks produced two detections ~0.7s apart.
Fixed: `Deduplicator` class in chirp_monitor.py suppresses detections within
(preamble_duration + block_duration) of the last. Verified on live beacon:
exactly one detection per 3s transmission.

### 2. Bin variance is high (~1000 bins spread)
De-chirped peak bins wander by ~1000 across a single preamble. Likely causes:
- 8x oversampling (1 MHz sample rate vs 125 kHz BW) means only 4096 of 32768
  bins are in the LoRa band — small alignment errors move peak a lot
- Crystal frequency offset (~60 kHz) between Pluto and RFM95W

`decimate_to_lora()` added to chirp_detect.py — FFT-based brick-wall filter +
downsample to 2*BW (250 kHz). Tested and working on synthetic signals.
NOT yet integrated into detection pipeline because the peak/median SNR metric
gives ~6 dB lower values at 8192 bins vs 32768 bins (fewer noise bins to
average over), causing detections to fall below threshold. Fix #3 (better SNR
metric) should resolve this. Decimation will be used for symbol demodulation.

### ~~3. Noise floor SNR metric~~ DONE
The old peak/median SNR gave ~11-12 dB on pure noise (max of N Rayleigh RVs
exceeds median by ~sqrt(log2(N))). Fixed by subtracting the expected noise
baseline: `corrected = raw - 10*log10(log2(N_bins))`. Now noise reads ~0 dB,
signal ~7-8 dB, with default threshold lowered to 5.0 dB (~7 dB margin vs
the old ~4 dB). Metric is consistent across FFT sizes (tested SF7 and SF12).

## Next Steps (in order)

### ~~Step 1: Symbol Demodulation~~ DONE
chirp_demod.py implements:
- `generate_lora_frame()` — synthetic frame generator for testing
- `find_sfd()` — locates SFD down-chirps via up/down energy comparison
- `extract_symbols()` — dechirp + FFT per symbol, full FFT gives direct
  bin-to-symbol mapping regardless of oversampling rate
- `demodulate()` — full pipeline: decimate → detect → find_sfd → extract

chirp_monitor.py `--demod` flag enables live symbol extraction (35-symbol
blocks to fit full frame). NOT YET TESTED ON LIVE BEACON — next step.

### Step 2: LoRa Protocol Decode
Convert raw symbols to bytes. This is the hard part — multiple layers:

1. **De-gray code** — LoRa gray-codes symbols to minimize bit errors
2. **De-interleave** — symbols are interleaved in blocks of (4+CR) across SF bits
3. **Hamming FEC decode** — each codeword is (4+CR) bits, CR=1 means (5,4) Hamming
4. **De-whiten** — XOR with LFSR pseudo-random sequence
5. **Header decode** — first 8 symbols use reduced rate (SF-2 bits), contain
   payload length and CR
6. **CRC check** — CRC-16 on payload

References for the exact algorithms:
- Matt Knight's talk "Decoding LoRa" (DEF CON 24)
- gr-lora by rpp0 (GNU Radio LoRa decoder, MIT license)
- https://github.com/tapparelj/gr-lora_sdr (another reference implementation)

Test approach: send "HELLO" from beacon, decode, verify output matches.

### Step 3: Decode "HELLO"
Integration test — full pipeline from raw IQ to decoded payload bytes.

Build chirp_decode.py that:
1. Reads IQ from SDR (sdr_source)
2. Detects preamble (chirp_detect)
3. Extracts symbols (step 1)
4. Decodes to bytes (step 2)
5. Prints decoded payload

This is the "it works" demo. Once we can decode "HELLO" from the beacon,
the Python prototype is complete and we can think about the C port.

## Architecture Notes

### File layout after completion
```
lora/
  sdr_source.py        — IQ from rtl_433 (done)
  chirp_detect.py      — preamble detection (done, needs dedup)
  chirp_demod.py       — symbol extraction from detected preamble (done)
  lora_decode.py       — NEW: gray/interleave/hamming/whiten/CRC
  chirp_waterfall.py   — visualization (done)
  chirp_monitor.py     — CLI preamble detector (done)
  chirp_decode.py      — NEW: full pipeline CLI
  tests/
    test_chirp_detect.py   (done)
    test_sdr_source.py     (done)
    test_chirp_demod.py    — test symbol extraction with synthetic signals (done)
    test_lora_decode.py    — NEW: test each decode stage with known vectors
```

### Key parameters confirmed from live testing
- CS16 format, normalize by /32768.0
- Noise floor SNR: ~0 dB (corrected peak/median metric)
- Real chirp SNR: ~7-8 dB
- Working threshold: 5 dB
- Frequency offset: ~60 kHz (bins ~30600-31200 out of 32768)
- Beacon interval: 3 seconds
- Preamble: 8 chirps = 262 ms at SF12
- Block size: 327680 samples (327.7 ms) works well
- Quarter-symbol overlap (step = sym_len//4) sufficient for alignment
