# LoRa Chirp Detector — Backlog

## Current Status (2026-03-07)

### What works
- **Preamble detection** — solid on live SF7/125k beacon (~10-12 dB SNR,
  6-17 chirps). Single-SF mode at 0.52 MSPS (52% real-time at 1 MSPS).
  46 tests passing in chirp_detect.
- **Symbol demod** — works on synthetic signals (9/11 tests pass).
  Full pipeline has constant +96 symbol offset from 0.25-symbol SFD
  alignment issue. Live demod produces garbage due to frequency
  offset / timing coupling at 8x oversampling.
- **Waterfall display** — works for visual confirmation.
- **Deduplication** — suppresses duplicate detections from preambles
  spanning block boundaries.

### What doesn't work
- **SF scan mode** — too slow for real-time in Python. 6 SFs at fixed
  125k BW gives ~0.07 MSPS (7% real-time). Giant blocks sized for SF12
  create hundreds of noise windows for small SFs, producing false runs
  with enormous bin_std. Score-based filtering helps but doesn't fix
  the fundamental speed problem.
- **BW auto-detection** — dropped. 18 combos was 5x slower than
  real-time. The "BW doubling" we saw was actually other LoRa devices
  in the ISM band, not a detection bug.
- **Live symbol extraction** — frequency offset and timing alignment
  are coupled at 8x oversampling. Quarter-symbol timing error shifts
  dechirp peak by ~1000 bins.

### Test summary
- chirp_detect: 46/46 pass
- chirp_demod: 9/11 pass (2 full-pipeline failures from SFD offset)
- chirp_monitor: 9/9 pass
- sdr_source: 7/7 pass
- Total: 71/73 pass

## Hardware
- Beacon: RFM95W on Arduino UNO, SF7-12 cycle or fixed SF12
- SDR: ADALM-Pluto via rtl_433, CS16 format, 1 MSPS
- Antenna: V-dipole, ~3.25" arms, 915 MHz tuned

## Architecture

```
lora/
  sdr_source.py        — IQ from rtl_433 subprocess (done)
  chirp_detect.py      — preamble detection + estimate_parameters (done)
  chirp_demod.py       — symbol extraction pipeline (partial)
  chirp_waterfall.py   — matplotlib waterfall display (done)
  chirp_monitor.py     — CLI detector with --scan, --demod (done)
  analyze_capture.py   — offline IQ analysis tool (done)
  sf_bw_detection.md   — experiment notes for SF/BW detection work
  arduino/
    lora_beacon_simple/    — fixed SF12/125k beacon
    lora_beacon_SF_cycle/  — cycles SF7-12 at 125k
  tests/
    test_chirp_detect.py   — 46 tests
    test_chirp_demod.py    — 11 tests (2 known failures)
    test_chirp_monitor.py  — 9 tests
    test_sdr_source.py     — 7 tests
```

## Key Technical Facts
- CS16 format: 4 bytes/sample, int16 I/Q, normalized by /32768.0
- Noise floor SNR: ~0 dB (corrected peak/median metric)
- Real chirp SNR: ~7-12 dB depending on SF and conditions
- Working threshold: 5 dB
- Beacon crystal offset: ~-30 to -50 kHz (bin ~950-1010 for SF7)
- 915 MHz ISM band has 4+ other LoRa devices nearby

## Performance Measurements
| Mode | MSPS | % real-time | Notes |
|------|------|-------------|-------|
| SF7 only | 0.52 | 52% | reliable detection every ~3s |
| SF scan (6 SFs) | ~0.07 | 7% | too slow, misses most blocks |
| BW+SF scan (18) | 0.31 | 31% | dropped — too slow + unnecessary |

## Known Issues

### 1. Python too slow for multi-SF real-time
Single SF at 1 MSPS is only 52% real-time. Multi-SF scan is infeasible
in Python. Options: lower sample rate, pyfftw, round-robin SFs, or C port.

### 2. Demod SFD alignment (+96 symbol offset)
The 2.25-symbol SFD means data starts at 0.25-symbol offset from
preamble alignment. Current extract_symbols doesn't account for this,
producing a constant +96 offset (for SF7, 96 = 128 * 0.75).

### 3. Demod frequency offset / timing coupling
At 8x oversampling, timing error of dt samples shifts dechirp peak
by (chirp_rate * dt / bin_spacing) bins. Must estimate freq offset
at the same sub-symbol alignment as data extraction.

### 4. chirp_demod.py has dead code
Unused scipy import, experimental functions (fine_align,
estimate_frequency_offset) that may not be needed in final approach.

## Experiment Log
See sf_bw_detection.md for detailed notes on BW detection experiments,
hypotheses tested, and performance measurements.
