# SF/BW/Frequency Auto-Detection — Progress Notes

## Goal
Detect spreading factor (SF), bandwidth (BW), and center frequency from live
LoRa signals without prior knowledge of the parameters.

## Status: WORKING CORRECTLY (46/46 tests, live validated)

The algorithm correctly identifies SF, BW, and frequency offset. What we
previously thought was "BW doubling" was actually correct detection of REAL
signals from other LoRa devices in the 915 MHz ISM band.

## Breakthrough: Multiple LoRa Devices in 915 MHz ISM Band

### Evidence (from ~200s live capture, 2025-03-07)
At least 4 distinct non-beacon LoRa signal sources detected:

| Source | SF/BW | bin (FFT) | Freq offset | Chirps | Repeats |
|--------|-------|-----------|-------------|--------|---------|
| A | SF7/250k | 54 | +106 kHz | 31 | ~60-80s |
| B | SF12/250k | 0-252 | 0-15 kHz | 16-26 | ~140s |
| C | SF9/250k | 1811 | -116 kHz | 8 | ~128s |
| D | SF10/500k | 121-2025 | varies | 12-24 | ~67s |

These are bin_std=0 detections — mathematically impossible to produce from
wrong-BW dechirp of our beacon signal. They are real LoRa devices (likely
LoRaWAN gateways or IoT sensors nearby).

Our beacon (SF7/125k) was only detected once at 57.2s:
  SF7/125k, score=14.0, chirps=17, bin_std=0.4, bin=970 (~-53 kHz crystal offset)

### Why "BW doubled" interpretation was wrong
We assumed all detections came from our beacon. Since beacon was configured for
125k BW, any 250k detection looked like a "2x BW error." In reality:
- SF12/125k from beacon → low SNR, not reliably detected (score < 0.3)
- SF12/250k from NEARBY DEVICE → high SNR, bin_std=0, score=21.2
- These are different signals, not the same signal misidentified

## Hypotheses Tested

### DISPROVEN: 2x Sample Rate Mismatch
Tested synthetic SF12/125k at 2 MSPS, analyzed at 1 MSPS → produces garbage
(wrong SF, score=0.5), NOT systematic 2x BW shift.

### DISPROVEN: Full-packet structure confuses scorer
Added sync words + SFD + data symbols to synthetic signals → still correctly
discriminated (46/46 tests pass).

### CONFIRMED: Other devices in ISM band
Live data shows multiple signals with bin_std=0 at various SF/BW combos that
don't match our beacon. These are real LoRa traffic.

## Scoring Function (working well)
```python
bin_std = det.get('bin_std', 0)
n = det['n_chirps']
chirp_factor = min(n, 12) / n
score = det['snr_db'] / (1 + bin_std / 5.0) * chirp_factor
```

## Processing Performance Note
With --scan (18 SF/BW combos), processing is ~5x slower than real-time.
Measured throughput: 0.185 MSPS at 1 MSPS SDR rate. This means we process
~1/5 of blocks, missing some signals. The SDR pipe blocks when buffer fills.
To improve: could limit candidates or parallelize.

## Diagnostics Available
- `--debug --scan` prints all 18 candidate scores per block
- `--iq-save DIR` saves .npy files on detection
- `--sample-rate RATE` overrides assumed rate
- `analyze_capture.py` for offline analysis of saved .npy files
- Measured data rate printed every 50 blocks

## Files Modified
- `chirp_detect.py` — estimate_parameters(), bin_to_freq_offset(), bin_std
  scoring, verbose param, generate_lora_packet()
- `chirp_monitor.py` — --scan, --debug, --iq-save, --sample-rate, data rate
- `lora_beacon/lora_beacon.ino` — Multi-config cycling
- `tests/test_chirp_detect.py` — 46 tests including full-packet, rate mismatch
- `analyze_capture.py` — Offline IQ analysis tool

## Decision: Drop BW Auto-Detection, Keep SF-Only

### Problem
BW scanning (3 BWs x 6 SFs = 18 combos) is too slow for real-time processing.
Measured ~0.31 MSPS after vectorizing the FFT loop (was 0.185 MSPS before),
still only 31% of the 1 MSPS SDR rate. Beacon was confirmed running (packet
1185+) but only detected once in 200s of scan-mode capture due to low duty cycle.

SF7-only mode (1 combo) ran at 0.52 MSPS and detected 155 signals in 42s,
including the beacon at bin ~990 (SNR ~10-12 dB, repeating every ~3s).

### Recommendation (agreed with user)
- **Fix BW at 125 kHz** (or make it a CLI param). Most LoRa deployments use
  125 kHz (LoRaWAN default). 250k/500k are rare.
- **Scan only SFs (7-12)** — 6 combos instead of 18, all at the same BW, so
  window sizes scale predictably.
- SF is the interesting variable for rtl_433 (tells range/data rate tradeoff).
- BW scanning can be added later if there's demand.

### Performance impact
- 6 combos instead of 18 → ~3x faster
- Combined with vectorized FFT (1.7x) and reduced offsets (2x) → should
  approach or exceed real-time for SF-only scan
- Could further optimize by sharing intermediate results across SFs

## Optimizations Applied
1. **Vectorized batch FFT** — reshape signal into 2D window array, single
   np.fft.fft() call per offset instead of per-window Python loop. 1.7x speedup.
2. **Reduced starting offsets** — 2 offsets (half-symbol step) instead of 4
   (quarter-symbol). Slight alignment tolerance loss, ~2x speedup.

## Other Findings
- Beacon confirmed running (multi-config firmware flashed, packet 1185+)
- SF7/125k beacon detected in non-scan mode: bin ~984-1011 (~-25 to -35 kHz
  crystal offset), SNR ~10-12 dB, repeating every ~3s matching beacon interval
- 915 MHz ISM band has significant LoRa traffic from other devices (at least
  4 distinct sources at SF7/250k, SF12/250k, SF9/250k, SF10/500k)

---

## Next Phase: SF-Only Scan Mode

### Objectives
1. **SF-only scanning** — `estimate_parameters()` scans SF 7-12 at a single
   fixed BW. No more BW enumeration (18 combos -> 6 combos).
2. **BW as a parameter** — Default 125 kHz, configurable via `--bw` CLI arg
   (like rtl_433 does). Not auto-detected.
3. **Batched FFTs** — All detection uses vectorized numpy batch FFT (no
   per-window Python loops). Already partially done, needs cleanup.
4. **Explicit real-time performance metric** — Print measured processing
   speed as a fraction/multiple of real-time (e.g. "0.8x real-time" or
   "1.2x real-time"). Goal: >= 1.0x for SF-only scan at 1 MSPS.
5. **100% beacon detection** — With the beacon running (SF7/125k, ~3s
   interval), every beacon transmission within the capture window should
   be detected. No missed beacons due to processing lag or scan overhead.
6. **All tests pass** — Existing 46 tests must continue to pass. Add new
   tests for SF-only scan path if needed.

### Done
- [x] Simplify `estimate_parameters()` to only iterate over SFs (not BWs)
- [x] Remove BW candidates list; use single `bw=` param (default 125k)
- [x] Batched FFT used everywhere (no per-window Python loops)
- [x] Real-time speed reporting (processing MSPS / SDR MSPS ratio)
- [x] Run full test suite — 46/46 pass
- [x] Update `--scan` help text to reflect SF-only behavior

### Failed
- [ ] Live test: 100% beacon detection — FAILED
  - SF-only scan at 0.51x real-time (still too slow)
  - Only 1 detection in 90s with score >= 3.0
  - Root cause: block sized for SF12 (327k samples) creates 320 windows
    for SF7 — noise windows form false runs with bin_std 300-900
  - Python numpy FFT is too slow for 6 SFs at 1 MSPS

### Conclusion
SF scan mode is not viable in Python at 1 MSPS. Even single-SF
detection runs at only 52% real-time. Multi-SF scan needs either:
- C implementation (the eventual rtl_433 goal)
- Lower sample rate
- Faster FFT library (pyfftw)
- Round-robin SF cycling (trade latency for throughput)
