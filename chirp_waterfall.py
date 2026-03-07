"""
LoRa Chirp Waterfall Display

Launches rtl_433 to stream raw IQ from the Pluto SDR and displays a live waterfall.
LoRa chirps will appear as diagonal lines sweeping across the bandwidth.

Usage:
  python chirp_waterfall.py
"""

import sys
import threading
import collections
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sdr_source import start_sdr, read_iq_blocks, DEFAULT_FREQ, DEFAULT_SAMPLE_RATE, DEFAULT_BANDWIDTH

# Waterfall settings
FFT_SIZE = 4096
WATERFALL_ROWS = 300
AVG_COUNT = 4
UPDATE_INTERVAL = 50  # ms


def compute_spectrum(samples, fft_size):
    """Compute power spectrum in dB from IQ samples."""
    window = np.hanning(fft_size)
    windowed = samples[:fft_size] * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
    return power_db


def reader_thread(proc, spectrum_queue, stop_event):
    """Continuously read IQ from rtl_433 stdout, average FFTs, and queue rows."""
    avg_buf = np.zeros(FFT_SIZE, dtype=np.float64)
    avg_count = 0

    for samples in read_iq_blocks(proc, block_size=FFT_SIZE):
        if stop_event.is_set():
            break

        spectrum = compute_spectrum(samples, FFT_SIZE)
        avg_buf += spectrum
        avg_count += 1

        if avg_count >= AVG_COUNT:
            spectrum_queue.append(avg_buf / avg_count)
            avg_buf = np.zeros(FFT_SIZE, dtype=np.float64)
            avg_count = 0

    stop_event.set()


def main():
    freq = DEFAULT_FREQ
    sample_rate = DEFAULT_SAMPLE_RATE

    print(f"Starting rtl_433: {freq}, {sample_rate/1e6:.0f} MHz sample rate")
    print("Launching SDR...")

    try:
        proc = start_sdr(freq, sample_rate, DEFAULT_BANDWIDTH)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("SDR streaming. Launching waterfall...")

    # Shared state
    spectrum_queue = collections.deque(maxlen=500)
    stop_event = threading.Event()

    reader = threading.Thread(target=reader_thread,
                              args=(proc, spectrum_queue, stop_event),
                              daemon=True)
    reader.start()

    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/sample_rate))
    freq_khz = freqs / 1e3

    # Initialize waterfall
    waterfall = np.full((WATERFALL_ROWS, FFT_SIZE), -200.0)

    # Set up plot
    fig, (ax_spec, ax_wf) = plt.subplots(2, 1, figsize=(10, 7),
                                          gridspec_kw={'height_ratios': [1, 3]})
    fig.suptitle(f'LoRa Chirp Waterfall - {freq}')

    spec_line, = ax_spec.plot(freq_khz, np.zeros(FFT_SIZE), color='cyan', linewidth=0.8)
    ax_spec.set_xlim(freq_khz[0], freq_khz[-1])
    ax_spec.set_ylim(-80, 0)
    ax_spec.set_ylabel('Power (dB)')
    ax_spec.set_xlabel('Offset from center (kHz)')
    ax_spec.grid(True, alpha=0.3)
    ax_spec.set_facecolor('black')

    extent = [freq_khz[0], freq_khz[-1], WATERFALL_ROWS, 0]
    wf_img = ax_wf.imshow(waterfall, aspect='auto', extent=extent,
                           cmap='inferno', vmin=-80, vmax=0,
                           interpolation='nearest')
    ax_wf.set_ylabel('Time (rows ago)')
    ax_wf.set_xlabel('Offset from center (kHz)')
    color_scaled = [False]

    plt.tight_layout()
    plt.ion()
    plt.show()

    print("Waterfall running. Close the window or Ctrl+C to stop.")

    try:
        while plt.fignum_exists(fig.number) and not stop_event.is_set():
            new_rows = []
            while spectrum_queue:
                new_rows.append(spectrum_queue.popleft())

            if new_rows:
                n = len(new_rows)
                if n >= WATERFALL_ROWS:
                    waterfall[:] = np.array(new_rows[-WATERFALL_ROWS:])
                else:
                    waterfall = np.roll(waterfall, -n, axis=0)
                    waterfall[-n:, :] = np.array(new_rows)

                # Auto-scale color range from actual data
                if not color_scaled[0]:
                    real_data = waterfall[waterfall > -200]
                    if len(real_data) > FFT_SIZE:
                        noise_floor = np.median(real_data)
                        vmin = noise_floor - 3
                        vmax = noise_floor + 30
                        wf_img.set_clim(vmin, vmax)
                        ax_spec.set_ylim(vmin, vmax)
                        color_scaled[0] = True
                        print(f"Auto-scaled: noise floor ~{noise_floor:.1f} dB, range [{vmin:.1f}, {vmax:.1f}]")

                spec_line.set_ydata(waterfall[-1, :])
                wf_img.set_data(waterfall)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(UPDATE_INTERVAL / 1000.0)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        proc.terminate()
        proc.wait(timeout=5)
        plt.close()
        print("Done.")


if __name__ == '__main__':
    main()
