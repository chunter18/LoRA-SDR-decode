# LoRa Preamble Detection — Findings

## Problem: Zero Detections on Live IQ

Our dechirp+FFT preamble detector (`detect_preamble` in `chirp_detect.py`)
returned zero detections on 10 seconds of saved IQ data (`diag_capture.npy`)
that contains three beacon transmissions. When we lowered thresholds, it would
occasionally fire but inconsistently.

## Root Cause #1: AGC Destroying the Signal

**This was the primary problem.** The Pluto SDR was running in AGC mode
(`slow_attack`) at ~71 dB gain, which completely destroyed the chirp signal.

### What AGC Does to LoRa

The Pluto SDR's AD9361 has automatic gain control that continuously adjusts
the RF gain to keep the ADC output near full-scale. With a strong LoRa
transmitter nearby:

1. **Between packets:** AGC cranks gain to ~71 dB, amplifying thermal noise
   to fill the entire ADC range (-5 dBFS). The spectrum appears uniformly
   loud across the full 1 MHz bandwidth.

2. **Packet arrives:** The strong chirp signal causes an AGC transient —
   gain drops 15-20 dB in a few milliseconds. Total power plummets from
   -5 dBFS to -25 dBFS.

3. **During the transient:** The gain is changing within/between symbols.
   Each chirp has a different amplitude, which destroys:
   - Dechirp correlation (amplitude envelope modulates the chirp)
   - Symbol-to-symbol autocorrelation (consecutive windows have different gains)

4. **After settling:** The preamble is already over. SF7 preamble = 8 chirps
   x 1.024 ms = 8.2 ms. The AGC transient lasts about that long.

### Evidence

With AGC (original `diag_capture.npy`):
- Noise floor: -5.8 dBFS (AGC amplifies noise to near full-scale)
- Signal: buried, only visible as power dips at beacon times
- Spectrum: flat across entire 1 MHz (noise dominates everywhere)
- Peak clipping: 0.8% of samples

With manual gain at 20 dB (`capture_manual_g20_10s.npy`):
- Noise floor: -41.3 dBFS (true thermal noise)
- Signal: -15.6 dBFS (25 dB above noise!)
- Three clean beacons visible at ~0s, ~3.5s, ~7.4s
- Zero clipping

### Why rtl_433 Works Fine with AGC

rtl_433 uses OOK/FSK pulse detection that measures signal presence and timing,
not phase-coherent chirp correlation. AGC naturally helps OOK by keeping
signals in the ADC's usable range. The pulse detector adapts its threshold
to the current gain level.

LoRa's dechirp+FFT requires stable amplitude across each 1+ ms symbol.
AGC gain changes within the symbol destroy the correlation.

## Root Cause #2: `rx_sdr` Gain Settings Don't Work

The `rx_sdr` command supports `-g GAIN` and `-t gain_mode=manual`, and both
report success in stderr:

```
Tuner gain set to 20.00 dB.
set key=|gain_mode|, value=|manual|
```

**But they have NO effect on the actual hardware.** Testing with `iio_attr`
after `rx_sdr` runs shows the Pluto stays in `slow_attack` mode at whatever
gain the AGC chose:

```
gain_control_mode value: slow_attack
hardwaregain value: 45.000000 dB
```

This appears to be a bug in the SoapyPlutoSDR driver — the `writeSetting()`
call that `-t` triggers doesn't map to `setGainMode()`, and the `-g` flag
sets a gain value that gets overridden by the AGC.

## How to Set Manual Gain

### Step 1: Set gain via `iio_attr` (before capturing)

```bash
# Set manual gain mode
"C:/Program Files/PothosSDR/bin/iio_attr.exe" -u "usb:1.32.5" \
    -i -c ad9361-phy voltage0 gain_control_mode manual

# Set gain (0-73 dB range)
"C:/Program Files/PothosSDR/bin/iio_attr.exe" -u "usb:1.32.5" \
    -i -c ad9361-phy voltage0 hardwaregain 20
```

**Gain settings do NOT persist across Pluto resets.** After a reset or
power cycle, the Pluto reverts to `slow_attack` AGC at 71 dB gain.
The USB address also changes (e.g., `usb:1.32.5` -> `usb:1.34.5`).
You must re-run `iio_attr` after every reset.

The USB address can change when the Pluto is reconnected.
Find it with:
```bash
"C:/Program Files/PothosSDR/bin/iio_info.exe" -s
```

### Step 2: Capture normally with `rx_sdr`

After `iio_attr` sets the gain, `rx_sdr` preserves it (it doesn't reset
the gain mode on connect):

```bash
rx_sdr -d driver=plutosdr -f 915000000 -s 1000000 -F CS16 -
```

### Step 3: Verify

```bash
"C:/Program Files/PothosSDR/bin/iio_attr.exe" -u "usb:1.32.5" \
    -i -c ad9361-phy voltage0 gain_control_mode
# Should print: manual

"C:/Program Files/PothosSDR/bin/iio_attr.exe" -u "usb:1.32.5" \
    -i -c ad9361-phy voltage0 hardwaregain
# Should print: 20.000000 dB
```

### Gain Selection Guide

| Scenario | Recommended Gain | Rationale |
|----------|-----------------|-----------|
| Transmitter inches away | 0-10 dB | Signal is extremely strong |
| Same room, few meters | 20-30 dB | Typical indoor testing |
| Different room / outdoors | 40-50 dB | Moderate path loss |
| Long range / weak signal | 60-73 dB | Maximum sensitivity |

**Rule of thumb:** Set gain so the signal peaks at -10 to -20 dBFS during
transmission. This leaves headroom for stronger signals while keeping the
signal well above the noise floor. If you see clipping (peak magnitude > 1.0),
reduce gain.

## Close-Range vs Distant Transmitter

### With transmitter inches away (current setup)

- Signal is 25+ dB above wideband noise at gain=20
- Signal durations observed: 549ms, 833ms, 833ms — much longer than a
  single SF7 "HELLO" packet (~46ms). Cause still under investigation.
- Dechirp+FFT fires on every block during signal presence (preamble AND data)
- Autocorrelation detector still fails on this data (max corr 0.34) —
  under investigation

### With transmitter at realistic range

- Lower SNR means the noise floor matters more
- AGC would be less destructive because the signal doesn't cause large gain
  swings (the AGC barely notices weak signals)
- At moderate distances (10-50 dB path loss), a gain of 40-60 dB with manual
  mode would be appropriate
- The preamble detection threshold (SNR or correlation) becomes critical

### AGC at realistic range

Ironically, AGC might work acceptably at realistic distances because:
1. The signal doesn't drive the AGC into rapid gain changes
2. The gain settles before/during the preamble rather than hunting
3. The AGC keeps the noise floor stable around -5 to -10 dBFS

The AGC problem is worst when the signal is strong enough to trigger
rapid gain changes but the preamble is short (low SF). At SF12 (preamble
= 262 ms), even aggressive AGC has time to settle.

## Additional Finding: Window-Chirp Misalignment

Even with correct gain, the dechirp+FFT detector has a secondary issue at
oversampled rates (1 MSPS for 125 kHz BW):

At 1 MSPS, the FFT has 1024 bins but only 128 (12.5%) are in-band. When a
window straddles two chirps, the dechirped signal splits into two tones. This
causes the peak bin to scatter across the full FFT range. The detector
compensates by looking for consecutive high-SNR windows regardless of bin
value, which effectively makes it a "LoRa signal present" detector rather
than a "preamble" detector.

For proper preamble-only detection, we would need either:
- Decimation to 1x rate (sample_rate = BW) and consecutive bin matching
- Symbol-spaced autocorrelation (only identical preamble chirps correlate)
- Sync word detection after coarse signal detection

## Reference Implementations

Studied two mature GNU Radio LoRa SDR implementations:

### gr-lora by rpp0 (github.com/rpp0/gr-lora)

Uses **Schmidl-Cox autocorrelation** for preamble detection:

```
autocorr = |sum(window1 * conj(window2))| / sqrt(E1 * E2)
```

Where window1 and window2 are two consecutive symbol-length windows. Since
preamble chirps are identical, consecutive windows are highly correlated
(autocorr -> 1.0). Threshold: 0.90.

After detection, uses instantaneous-frequency cross-correlation for fine sync
and SFD detection. Dechirp+FFT is only used for demodulation AFTER alignment.

### gr-lora_sdr by tapparelj (github.com/tapparelj/gr-lora_sdr)

Decimates to 1x rate (sample_rate = BW) before all FFT processing. At 1x rate,
each symbol is exactly 2^SF samples and the FFT bins correspond directly to LoRa
symbol values.

Detection uses **consecutive bin matching**: dechirp each symbol-length window,
FFT, argmax. If consecutive windows produce the same bin (within +/-1),

increment counter. Trigger when counter reaches preamble_length - 3.

**Neither implementation uses an SNR threshold for detection.** SNR is
informational only. Frame validation uses sync word matching.

### Key Design Differences from Our Approach

| Aspect | Our code | gr-lora (rpp0) | gr-lora_sdr (tapparelj) |
|--------|----------|----------------|-------------------------|
| Detection method | Dechirp+FFT, SNR runs | Autocorrelation | Dechirp+FFT, bin matching |
| Processing rate | SDR rate (1 MSPS) | Any (channelized) | 1x rate (= BW) |
| Detection gate | SNR > threshold | Autocorr > 0.90 | Consecutive bin match |
| Alignment handling | Multiple offsets | Doesn't need alignment | Doesn't need (at 1x rate) |
