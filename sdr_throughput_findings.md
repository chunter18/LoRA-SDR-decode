# SDR IQ Throughput: rtl_433 vs rx_sdr

## Problem

At 1 MSPS sample rate from the Pluto SDR, rtl_433 piped IQ at only **0.6 MSPS**
(60% real-time). This caused the LoRa chirp detector to operate on data with an
effective sample rate mismatch — the reference chirp was generated for 1024
samples/symbol but the actual data had ~614 samples per real symbol. This
produced false/duplicate detections from a broken correlator.

## Root Cause

rtl_433 runs its full signal processing pipeline on every sample even when using
`-w -` (raw IQ dump mode). The pipeline is single-threaded and blocking.

### What rtl_433 does per buffer (even with `-w -`)

In `rtl_433.c`, having a dumper (`-w -`) forces the full demod pipeline:

```c
int process_frame = demod->squelch_offset <= 0 || !noise_only
                    || demod->load_info.format || demod->analyze_pulses
                    || demod->dumper.len || demod->samp_grab;
```

The processing steps, all applied to every sample:

1. **Magnitude/envelope detection** (`baseband.c`) — AM demod, ~10-15 ops/sample
2. **Low-pass filtering** (`baseband.c`) — baseband filter, ~4-6 ops/sample
3. **FM demodulation** (`baseband.c`) — atan2/phase unwrap, expensive
4. **Pulse detection** (`pulse_detect.c`) — OOK/FSK edge detection, state machine
5. **Device decoders** — disabled with `-R 0`, but steps 1-4 still run
6. **Format conversion** — per-sample int16 scaling loop for CS16 output
7. **fwrite() to stdout** — buffered I/O

Steps 1-4 are pure waste when all you want is raw IQ. The callback is blocking
in a single-threaded mongoose event loop:

```c
if (ev->ev == SDR_EV_DATA) {
    sdr_callback((unsigned char *)ev->buf, ev->len, cfg);  // BLOCKING
}
```

If the pipeline takes longer than the buffer interval, rtl_433 can't keep up.

## Measurements

### Raw pipe throughput (no SDR)

Python-to-Python pipe on Windows: **360 MB/s** (90 MSPS equivalent).
The pipe itself is not a bottleneck.

### rtl_433 `-R 0 -w -` at 1 MSPS

Raw read from rtl_433 stdout, no processing on the Python side:
**0.60 MSPS** (60% real-time).

### rx_sdr (SoapySDR raw streamer) at 1 MSPS

Raw read from rx_sdr stdout, no processing on the Python side:
**1.01 MSPS** (100% real-time).

### Python chirp detection processing speed

Pure numpy `detect_preamble()` on pre-loaded data (no I/O):
**10-13 MSPS** (10-13x real-time). Processing is not the bottleneck.

### End-to-end with detection

| Backend  | Throughput | Detection quality |
|----------|-----------|-------------------|
| rtl_433  | 0.53 MSPS | ~35 detections/30s (many false/duplicate) |
| rx_sdr   | 1.00 MSPS | ~9 detections/30s (1 per 3s beacon, correct) |

## Solution

Switched default SDR backend from rtl_433 to `rx_sdr`
(`C:\Program Files\PothosSDR\bin\rx_sdr.exe`). This is a lightweight SoapySDR
raw IQ streamer that does zero DSP — just reads from the SDR and writes to
stdout. It uses the same SoapySDR/SoapyPlutoSDR/libiio/libusb stack as rtl_433,
so no new dependencies.

Command used:
```
rx_sdr -d driver=plutosdr -f 915000000 -s 1000000 -F CS16 -b 1000000 -
```

The old rtl_433 backend is still available via `--backend rtl_433` for cases
where its protocol decoders are needed.

## Other things tried that didn't help

- **Larger blocks** (10 → 100 symbols): no effect because I/O was the bottleneck
- **Vectorized batch FFT**: no effect for same reason (processing was already fast)
- **Optimized pipe reads** (single read vs 64KB chunks): no effect
- **Lower sample rate** (521 kHz Pluto minimum): non-power-of-2 FFT issues, still 0.5x
- **Direct libiio from Python** (ctypes): PothosSDR's libiio 0.19 network backend
  can't create streaming buffers (USB works but isn't exposed to Python).
  SoapyPlutoSDR connects via USB/libusb which is why it works fine.

## libiio note

PothosSDR bundles libiio 0.19 which only has `xml`, `ip`, and `usb` backends.
The `usb` backend works (SoapyPlutoSDR uses it), but Python ctypes can only
access the `ip` backend, which has a buffer creation bug in 0.19
("Open unlocked: -32"). Installing a newer libiio risks breaking the existing
PothosSDR/SoapySDR/rtl_433 stack since they're linked against the bundled DLL.
