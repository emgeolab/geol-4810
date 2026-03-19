#!/usr/bin/env python3
"""Batch segment ASCII time series and compute complex FFT coefficients per segment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read ASCII time-series files, split each series into k segments, "
            "and compute one set of complex FFT coefficients per segment."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing ASCII files. Default: current directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ascii",
        help="Glob pattern for input files. Default: *.ascii",
    )
    parser.add_argument(
        "-k",
        "--segments",
        type=int,
        required=True,
        help="Number of segments to split each time series into.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Sampling interval in seconds. Default: 0.5",
    )
    parser.add_argument(
        "--window",
        choices=("none", "hann"),
        default="hann",
        help="Window applied before FFT. Default: hann",
    )
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="Remove the mean from each segment before FFT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spectra_output"),
        help="Directory for CSV outputs. Default: spectra_output",
    )
    return parser.parse_args()


def load_series(path: Path) -> np.ndarray:
    data = np.loadtxt(path, ndmin=1)
    if data.ndim > 1:
        if 1 in data.shape:
            data = np.ravel(data)
        else:
            raise ValueError(
                f"{path.name} has multiple columns. This script expects one value per sample."
            )
    if data.size == 0:
        raise ValueError(f"{path.name} is empty.")
    return data.astype(float, copy=False)


def split_into_segments(series: np.ndarray, k: int) -> list[np.ndarray]:
    if k <= 0:
        raise ValueError("segments must be a positive integer.")
    if k > series.size:
        raise ValueError(
            f"Cannot split {series.size} samples into {k} non-empty segments."
        )

    segment_length = series.size // k
    if segment_length < 2:
        raise ValueError(
            f"Each segment must contain at least 2 samples; got length {segment_length}."
        )

    trimmed_length = segment_length * k
    trimmed = series[:trimmed_length]
    return [trimmed[i * segment_length : (i + 1) * segment_length] for i in range(k)]


def build_window(length: int, name: str) -> np.ndarray:
    if name == "none":
        return np.ones(length, dtype=float)
    if name == "hann":
        return np.hanning(length)
    raise ValueError(f"Unsupported window: {name}")


def compute_spectrum(segment: np.ndarray, dt: float, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = segment.size
    windowed = segment * window
    fft_values = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, fft_values


def write_segment_spectrum(
    output_path: Path,
    freqs: np.ndarray,
    fft_values: np.ndarray,
) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "fft_real", "fft_imag"])
        writer.writerows(zip(freqs, fft_values.real, fft_values.imag))


def write_summary(
    output_path: Path,
    freq_matrix: np.ndarray,
    fft_matrix: np.ndarray,
) -> None:
    mean_fft = fft_matrix.mean(axis=0)
    std_fft_real = fft_matrix.real.std(axis=0, ddof=0)
    std_fft_imag = fft_matrix.imag.std(axis=0, ddof=0)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frequency_hz",
                "mean_fft_real",
                "mean_fft_imag",
                "std_fft_real",
                "std_fft_imag",
            ]
        )
        writer.writerows(
            zip(
                freq_matrix[0],
                mean_fft.real,
                mean_fft.imag,
                std_fft_real,
                std_fft_imag,
            )
        )


def process_file(
    path: Path,
    k: int,
    dt: float,
    window_name: str,
    detrend: bool,
    output_dir: Path,
) -> None:
    series = load_series(path)
    segments = split_into_segments(series, k)
    window = build_window(len(segments[0]), window_name)

    file_output_dir = output_dir / path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    freq_rows = []
    fft_rows = []

    for index, segment in enumerate(segments, start=1):
        if detrend:
            segment = segment - np.mean(segment)

        freqs, fft_values = compute_spectrum(segment, dt, window)
        freq_rows.append(freqs)
        fft_rows.append(fft_values)

        segment_path = file_output_dir / f"segment_{index:03d}_spectrum.csv"
        write_segment_spectrum(segment_path, freqs, fft_values)

    freq_matrix = np.vstack(freq_rows)
    fft_matrix = np.vstack(fft_rows)
    write_summary(file_output_dir / "summary_spectrum.csv", freq_matrix, fft_matrix)

    metadata_path = file_output_dir / "metadata.txt"
    samples_used = len(segments) * len(segments[0])
    samples_dropped = series.size - samples_used
    metadata_path.write_text(
        "\n".join(
            [
                f"input_file={path.name}",
                f"total_samples={series.size}",
                f"segments={k}",
                f"segment_length={len(segments[0])}",
                f"samples_used={samples_used}",
                f"samples_dropped={samples_dropped}",
                f"dt_seconds={dt}",
                f"sampling_rate_hz={1.0 / dt}",
                f"window={window_name}",
                f"detrend_mean={'yes' if detrend else 'no'}",
            ]
        )
        + "\n"
    )

    print(
        f"Processed {path.name}: {k} segments, "
        f"{len(segments[0])} samples/segment, dropped {samples_dropped} samples."
    )


def main() -> int:
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' in {input_dir}."
        )

    for path in files:
        process_file(
            path=path,
            k=args.segments,
            dt=args.dt,
            window_name=args.window,
            detrend=args.detrend,
            output_dir=output_dir,
        )

    print(f"All outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
