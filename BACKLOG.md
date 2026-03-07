# LoRa Chirp Detector — Backlog

## Current Status
- Preamble detection WORKS on live SF12 beacon (~19 dB SNR, 6-9 chirps per detection)
- Beacon: RFM95W on Arduino, SF12/125kHz, "HELLO" every 3 seconds
- SDR: Pluto via rtl_433 -w - (CS16 format, 1 MHz sample rate)
- 26/26 tests passing

## Known Issues to Fix First

### 1. Deduplicate detections
Preambles spanning two blocks produce two detections ~0.7s apart.
Fix: track last detection timestamp, suppress if within one preamble duration
(~262ms for SF12). Do this in chirp_monitor.py, not chirp_detect.py.

### 2. Bin variance is high (~1000 bins spread)
De-chirped peak bins wander by ~1000 across a single preamble. Likely causes:
- 8x oversampling (1 MHz sample rate vs 125 kHz BW) means only 4096 of 32768
  bins are in the LoRa band — small alignment errors move peak a lot
- Crystal frequency offset (~60 kHz) between Pluto and RFM95W
- Possible: need to decimate to LoRa bandwidth before de-chirp+FFT

Consider: bandpass filter around the signal, then decimate to ~250 kHz before
de-chirping. This would give 4096/(250k/125k) = cleaner bins and also make
FFTs much faster (4096 bins instead of 32768 for SF12).

### 3. Noise floor SNR metric
The median-based SNR gives ~11-12 dB on pure noise, which is a known property
of the metric (peak of N random variables exceeds median by ~11 dB for large N).
Current threshold of 15 dB works but leaves only ~4 dB margin above noise.
Consider using a better estimator (e.g., mean + k*std of magnitude bins).

## Next Steps (in order)

### Step 1: Symbol Demodulation
After detecting a preamble, extract the data symbols that follow it.

What to build:
- After preamble detection, identify the sync words (2 down-chirps that mark
  end of preamble / start of data)
- For each subsequent symbol period, de-chirp + FFT to get the symbol value
  (the FFT peak bin, mapped to 0..2^SF-1)
- Return list of raw symbol values

Key challenge: need to find exact symbol boundary. The preamble detection
gives approximate alignment (quarter-symbol precision). May need to refine
by trying sub-sample offsets and maximizing peak sharpness.

Test approach: transmit known payload "HELLO" repeatedly, verify we get
consistent raw symbols.

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
  chirp_demod.py       — NEW: symbol extraction from detected preamble
  lora_decode.py       — NEW: gray/interleave/hamming/whiten/CRC
  chirp_waterfall.py   — visualization (done)
  chirp_monitor.py     — CLI preamble detector (done)
  chirp_decode.py      — NEW: full pipeline CLI
  tests/
    test_chirp_detect.py   (done)
    test_sdr_source.py     (done)
    test_chirp_demod.py    — NEW: test symbol extraction with synthetic signals
    test_lora_decode.py    — NEW: test each decode stage with known vectors
```

### Key parameters confirmed from live testing
- CS16 format, normalize by /32768.0
- Noise floor SNR: ~11-12 dB (median-based metric)
- Real chirp SNR: ~17-21 dB
- Working threshold: 15 dB
- Frequency offset: ~60 kHz (bins ~30600-31200 out of 32768)
- Beacon interval: 3 seconds
- Preamble: 8 chirps = 262 ms at SF12
- Block size: 327680 samples (327.7 ms) works well
- Quarter-symbol overlap (step = sym_len//4) sufficient for alignment
