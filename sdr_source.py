"""
SDR IQ Source — reads raw IQ samples from SDR via subprocess.

Two backends available:
  - rx_sdr: lightweight raw IQ streamer via SoapySDR (default, full-rate)
  - rtl_433: protocol decoder with IQ passthrough (slower due to internal processing)

Both stream CS16 (signed 16-bit interleaved I/Q) on stdout.
This module launches the process and provides a generator for IQ blocks.
"""

import subprocess
import sys
import numpy as np

# Default SDR settings
RTL_433 = r"C:\Program Files\PothosSDR\bin\rtl_433-rtlsdr-soapysdr.exe"
RX_SDR = r"C:\Program Files\PothosSDR\bin\rx_sdr.exe"
DEFAULT_FREQ = "915M"
DEFAULT_SAMPLE_RATE = 1000000
DEFAULT_BANDWIDTH = 1000000

# CS16 format: 2 bytes I + 2 bytes Q = 4 bytes per sample
BYTES_PER_SAMPLE = 4


def parse_cs16(buf):
    """Parse CS16 bytes into complex64 numpy array.

    Args:
        buf: bytes object containing CS16 data (int16 I, int16 Q, ...).
             Length must be a multiple of 4.

    Returns:
        Complex64 numpy array of IQ samples, normalized to [-1, 1].
    """
    if len(buf) % BYTES_PER_SAMPLE != 0:
        raise ValueError(f"Buffer length {len(buf)} is not a multiple of {BYTES_PER_SAMPLE}")
    raw = np.frombuffer(buf, dtype=np.int16)
    iq = raw.astype(np.float32) / 32768.0
    return iq[0::2] + 1j * iq[1::2]


def start_sdr(freq=DEFAULT_FREQ, sample_rate=DEFAULT_SAMPLE_RATE,
              bandwidth=DEFAULT_BANDWIDTH, backend='rx_sdr'):
    """Launch an SDR subprocess streaming raw IQ to stdout.

    Args:
        freq: Center frequency string (e.g. "915M") or int in Hz.
        sample_rate: Sample rate in Hz.
        bandwidth: Analog bandwidth in Hz.
        backend: 'rx_sdr' (default, fast) or 'rtl_433' (slower, has decoder).

    Returns:
        subprocess.Popen process handle. Read IQ from proc.stdout.
    """
    freq_hz = str(int(float(str(freq).replace('M', 'e6').replace('k', 'e3'))))

    if backend == 'rx_sdr':
        cmd = [
            RX_SDR,
            "-d", "driver=plutosdr",
            "-f", freq_hz,
            "-s", str(int(sample_rate)),
            "-F", "CS16",
            "-b", str(int(sample_rate)),  # ~1 second output buffer
            "-",
        ]
    elif backend == 'rtl_433':
        cmd = [
            RTL_433,
            "-d", "driver=plutosdr",
            "-f", str(freq),
            "-s", str(sample_rate),
            "-t", f"bandwidth={bandwidth}",
            "-R", "0",
            "-w", "-",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend!r} (use 'rx_sdr' or 'rtl_433')")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check it didn't immediately fail
    try:
        ret = proc.wait(timeout=2)
        stderr = proc.stderr.read().decode(errors='replace')
        raise RuntimeError(
            f"{backend} exited with code {ret}\n{stderr[-500:]}"
        )
    except subprocess.TimeoutExpired:
        pass  # still running, good

    return proc


def read_iq_blocks(proc, block_size=4096):
    """Generator that yields complex64 IQ blocks from SDR stdout.

    Args:
        proc: subprocess.Popen from start_sdr().
        block_size: Number of IQ samples per block.

    Yields:
        Complex64 numpy arrays of length block_size.
    """
    chunk_bytes = block_size * BYTES_PER_SAMPLE

    while True:
        data = proc.stdout.read(chunk_bytes)
        if not data:
            break
        if len(data) < chunk_bytes:
            # Short read — accumulate until we have a full block
            buf = bytearray(data)
            while len(buf) < chunk_bytes:
                more = proc.stdout.read(chunk_bytes - len(buf))
                if not more:
                    return
                buf.extend(more)
            data = bytes(buf)
        yield parse_cs16(data)
