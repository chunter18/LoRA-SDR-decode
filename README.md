# LoRa Chirp Detector

A Python prototype for detecting LoRa signals via SDR, with the goal of
eventually contributing chirp demodulation support to
[rtl_433](https://github.com/merbanan/rtl_433).

## Background

LoRa uses Chirp Spread Spectrum (CSS) modulation, which is fundamentally
different from the OOK/FSK that rtl_433 currently handles. Each LoRa symbol
is a frequency chirp sweeping across the channel bandwidth. Data is encoded
in the chirp's starting frequency (cyclic shift).

Detection approach:
1. **De-chirp** — multiply received signal by conjugate of reference chirp
2. **FFT** — the chirp collapses to a single spectral peak
3. **Preamble detection** — look for repeated peaks at consistent intervals

This gives massive processing gain (up to 36 dB at SF12), allowing detection
of signals well below the noise floor.

## Hardware

- **SDR:** ADALM-Pluto (connected via USB, driven by rtl_433 + SoapySDR)
- **Transmitter (for testing):** RFM95W (SX1276) on Arduino — see `lora_beacon.ino`
- **Frequency:** 915 MHz (US ISM band)

## Quick Start

### 1. Waterfall Display

See live spectrum. LoRa chirps appear as diagonal lines.

```
cd lora
python chirp_waterfall.py
```

### 2. Chirp Monitor

Headless detection — prints LoRa preamble detections to stdout.

```
python chirp_monitor.py                  # scan SF7-12
python chirp_monitor.py --sf 12          # SF12 only
python chirp_monitor.py --threshold 8    # lower SNR threshold
```

### 3. Test Beacon

Flash `lora_beacon.ino` to an Arduino with an RFM95W module. Transmits
"HELLO" at SF12/125kHz every 3 seconds. See wiring notes in the file.

## Modules

| File | Purpose |
|------|---------|
| `sdr_source.py` | IQ source — launches rtl_433, reads CS16, yields complex samples |
| `chirp_detect.py` | Core DSP — chirp generation, de-chirping, FFT, preamble detection |
| `chirp_waterfall.py` | Live waterfall display (matplotlib) |
| `chirp_monitor.py` | CLI detection tool (no GUI) |
| `lora_beacon.ino` | Arduino test transmitter sketch |

## Tests

All tests use synthetic IQ data — no SDR hardware needed.

```
python -m pytest lora/tests/ -v -p no:asyncio
```

The `-p no:asyncio` flag avoids a plugin conflict on this system.

## Roadmap

1. **Python prototype** (current) — prove the algorithm with known beacon
2. **Symbol demodulation** — extract data symbols from detected packets
3. **LoRa protocol decode** — de-gray, de-interleave, Hamming FEC, CRC
4. **C port** — translate to C for rtl_433 integration
5. **PR upstream** — contribute chirp demodulator to rtl_433

## Technical Notes

- rtl_433 streams CS16 (signed 16-bit IQ) via `-w -` on stdout
- Pluto's native format is CS16 with full scale 2048 (12-bit ADC in 16-bit container)
- We normalize to [-1, 1] by dividing by 32768
- De-chirp uses conjugate multiplication: `received * conj(ref_up)` produces DC for
  unmodulated preamble chirps, or a tone at frequency proportional to symbol value
- FFT bin = symbol value directly when `n_fft = symbol_samples`
- Sample rate must be >= 521 kHz (Pluto AD9364 minimum); we use 1 MHz
