# LoRa Chirp Detector

A Python prototype for detecting LoRa preambles via SDR, with the goal of
eventually contributing chirp demodulation support to
[rtl_433](https://github.com/merbanan/rtl_433).

## Current Status

Preamble detection works reliably in real-time on live IQ from a Pluto SDR.
Tested at SF12/125kHz against an RFM95W beacon transmitting every 3 seconds:
**84% detection rate over 2 minutes, zero false positives.** Robust to clock
drift and BW mismatch between transmitter and receiver (~0.6% / 6400 ppm).

37 unit tests pass (synthetic data, no hardware needed).

## Background

LoRa uses Chirp Spread Spectrum (CSS) modulation, which is fundamentally
different from the OOK/FSK that rtl_433 currently handles. Each LoRa symbol
is a frequency chirp sweeping across the channel bandwidth. Data is encoded
in the chirp's starting frequency (cyclic shift). The preamble is 8+
identical unmodulated up-chirps.

## How the Detector Works

### De-chirp + FFT

The core idea: multiply the received signal by the conjugate of a reference
up-chirp. This "de-chirps" the signal — a received chirp collapses from a
wideband sweep into a single narrowband tone. An FFT then reveals this tone
as a sharp spectral peak.

```
received signal       reference (conj)       de-chirped result
  /  /  /  /    x    \  \  \  \  \    =     ---- (DC tone)
 /  /  /  /           \  \  \  \  \
```

For an unmodulated preamble chirp, the de-chirped tone lands at DC (bin 0).
For a data symbol with cyclic shift k, the tone lands at bin k. This gives
massive processing gain — up to 36 dB at SF12 — allowing detection of
signals below the noise floor.

### Preamble Detection (bin-match approach)

The detector (`detect_preamble_binmatch`) works in these steps:

1. **Slice into symbol-length windows.** The IQ stream is divided into
   non-overlapping windows of 2^SF samples (32,768 at SF12). Multiple
   starting offsets are tried (default 2) to handle arbitrary packet
   alignment.

2. **Batch de-chirp + FFT.** All windows are multiplied by the conjugate
   reference chirp and FFT'd in a single vectorized operation. For each
   window, the peak bin and SNR are recorded.

3. **Find high-SNR runs.** Consecutive windows with SNR above threshold
   (default 10 dB) indicate signal presence. These runs contain the full
   packet: preamble + sync word + data symbols.

4. **Separate preamble from data via bin-drift consistency.** Within each
   high-SNR run, preamble symbols all have the same de-chirp bin (plus a
   small linear drift from clock mismatch). Data symbols have random bins.
   The detector finds the longest prefix where consecutive bin differences
   have low variance (std < 50). This cleanly separates preamble from data
   without needing to know the exact bin value.

5. **Deduplicate across offsets.** Detections from different starting
   offsets that land within one symbol duration of each other are merged,
   keeping the highest-SNR result.

### Handling Clock Mismatch

Real transmitters and receivers have slightly different clocks. The RFM95W's
actual bandwidth is ~124,210 Hz vs the nominal 125,000 Hz (0.6% error from
SX1276 internal RC filter calibration). This causes the de-chirp bin to
drift linearly across preamble symbols (~25 bins/symbol at 1 MSPS).

The detector handles this by looking for **consistent** bin diffs rather than
**zero** bin diffs. A linear drift (constant bin-to-bin change) is preamble;
random jumps are data. The `max_bin_drift_std` parameter controls tolerance
(default 50 handles up to ~1% BW mismatch).

### Sliding Window (chirp_monitor.py)

The CLI monitor reads IQ in a sliding window: a 26-symbol analysis window
advances by 8 symbols (262 ms) per step. This ensures every packet's
preamble is fully contained in at least one window. A sample-position-based
deduplicator (not wall-clock time) suppresses repeated detections of the
same packet across overlapping windows.

Processing budget: ~70 ms per 262 ms step (27% of real-time at SF12).

### SNR Metric

De-chirp SNR is `20*log10(peak/median) - 10*log10(log2(N_bins))`. The
second term subtracts the expected max/median ratio for Rayleigh-distributed
noise, so pure noise reads ~0 dB regardless of FFT size. Real signals
typically read 7-12 dB at moderate distances, 25-30 dB close-range.

## Hardware

- **SDR:** ADALM-Pluto (AD9364, connected via USB, driven by rx_sdr + SoapySDR)
- **Transmitter (for testing):** RFM95W (SX1276) on Arduino — see `arduino/`
- **Frequency:** 915 MHz (US ISM band)
- **Sample rate:** 1 MSPS (Pluto minimum ~521 kHz)

### Pluto SDR Gain

The Pluto must be set to manual gain before capture. AGC destroys chirp
signals by changing gain within/between symbols (see `detection_findings.md`).

**Gain does not persist across resets.** After power-cycle or USB reconnect,
the Pluto reverts to AGC at 71 dB. The USB address also changes.

```bash
# Find Pluto USB address
"C:/Program Files/PothosSDR/bin/iio_info.exe" -s

# Set manual gain (run before every capture session)
"C:/Program Files/PothosSDR/bin/iio_attr.exe" -u "usb:X.Y.Z" \
    -i -c ad9361-phy voltage0 gain_control_mode manual
"C:/Program Files/PothosSDR/bin/iio_attr.exe" -u "usb:X.Y.Z" \
    -i -c ad9361-phy voltage0 hardwaregain 20
```

## Quick Start

### 1. Chirp Monitor

Headless detection — prints LoRa preamble detections to stdout.

```
cd lora
python chirp_monitor.py --sf 12          # SF12 only (recommended)
python chirp_monitor.py --sf 12 --threshold 8
python chirp_monitor.py --sf 7-12        # scan all SFs (experimental)
```

### 2. Waterfall Display

See live spectrum. LoRa chirps appear as diagonal lines.

```
python chirp_waterfall.py
```

### 3. Test Beacon

Flash `arduino/lora_beacon_simple/lora_beacon_simple.ino` to an Arduino with
an RFM95W module. Transmits "HELLO" at SF12/125kHz every 3 seconds. See
wiring notes in the file.

## Modules

| File | Purpose |
|------|---------|
| `chirp_detect.py` | Core DSP: chirp generation, de-chirp, FFT, preamble detection |
| `chirp_monitor.py` | CLI detection tool with sliding window and deduplication |
| `sdr_source.py` | IQ source: launches rx_sdr, reads CS16, yields complex samples |
| `chirp_waterfall.py` | Live waterfall spectrogram display (matplotlib) |
| `chirp_demod.py` | Symbol extraction (synthetic only, not yet validated live) |
| `detection_findings.md` | Detailed technical findings: AGC, gain, BW mismatch |
| `arduino/` | Arduino beacon sketches (simple + SF cycle) |

### Key Functions (chirp_detect.py)

| Function | Description |
|----------|-------------|
| `generate_chirp(params, direction)` | Generate reference up/down chirp |
| `dechirp_and_fft(samples, ref)` | De-chirp + FFT, returns peak bin and SNR |
| `detect_preamble_binmatch(samples, params)` | Main detector: multi-offset, drift-tolerant |
| `detect_preamble(samples, params)` | Simpler detector: SNR-run based (no bin matching) |
| `decimate_to_lora(samples, params)` | FFT-based decimation to near-BW rate |

## Tests

All tests use synthetic IQ data — no SDR hardware needed.

```
python -m pytest lora/tests/ -v -p no:asyncio
```

37 tests covering: chirp generation, de-chirp, FFT, preamble detection
(clean, noisy, frequency offset, arbitrary alignment, BW mismatch,
multi-packet), decimation, and parameter validation.

## Roadmap

1. ~~Python prototype~~ — preamble detection working
2. **Symbol demodulation** — validate on live signals (synthetic working)
3. **LoRa protocol decode** — de-gray, de-interleave, Hamming FEC, CRC
4. **C port** — translate to C for rtl_433 integration
5. **PR upstream** — contribute chirp demodulator to rtl_433

## Technical Notes

- rx_sdr streams CS16 (signed 16-bit IQ) to stdout at 1 MSPS
- Pluto's 12-bit ADC values are in 16-bit containers; we normalize by /32768
- De-chirp: `received * conj(ref_up)` = DC for preamble, tone at bin k for symbol k
- RFM95W actual BW is ~124,210 Hz (not 125,000) due to SX1276 RC filter cal
- ~30-50 kHz carrier offset between Pluto and RFM95W crystals (normal)
- The 915 MHz ISM band has no other LoRa transmitters in our test environment
