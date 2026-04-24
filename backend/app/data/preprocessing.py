from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt, resample_poly

from app.core.config import settings
from app.data.labels import quality_label_from_id


@dataclass
class PreprocessingResult:
    signal: np.ndarray
    sampling_rate: int
    r_peaks: np.ndarray
    beat_segments: np.ndarray
    fixed_segments: np.ndarray
    quality_id: int
    quality_label: str
    snr_db: float


def to_numpy_signal(
    signal: list[float] | np.ndarray | list[list[float]] | np.ndarray,
) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # For multi-lead input shape [lead, time], convert to reference lead.
        if arr.shape[0] <= arr.shape[1]:
            return arr[0].astype(np.float32)
        return arr[:, 0].astype(np.float32)
    raise ValueError("signal must be a 1D or 2D array-like object")


def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return signal.astype(np.float32)
    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError("sampling rates must be positive")
    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd
    return resample_poly(signal, up=up, down=down).astype(np.float32)


def butter_bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low_hz: float = settings.bandpass_low_hz,
    high_hz: float = settings.bandpass_high_hz,
    order: int = settings.bandpass_order,
) -> np.ndarray:
    nyquist = 0.5 * fs
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def remove_baseline_wander(signal: np.ndarray, fs: int) -> np.ndarray:
    # Two-stage median filtering keeps QRS morphology while removing slow drift.
    win1 = int(0.2 * fs)
    win2 = int(0.6 * fs)
    win1 = max(win1 + (1 - win1 % 2), 3)  # force odd
    win2 = max(win2 + (1 - win2 % 2), 3)
    baseline = medfilt(signal, kernel_size=win1)
    baseline = medfilt(baseline, kernel_size=win2)
    return (signal - baseline).astype(np.float32)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std < 1e-8:
        std = 1.0
    return ((signal - mean) / std).astype(np.float32)


def pan_tompkins_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    if signal.size < fs:
        return np.array([], dtype=np.int64)

    derivative = np.ediff1d(signal, to_begin=0.0)
    squared = derivative**2
    window = max(int(0.15 * fs), 1)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    distance = max(int(0.2 * fs), 1)
    threshold = float(np.mean(integrated) + 0.6 * np.std(integrated))
    peaks, _ = find_peaks(integrated, distance=distance, height=threshold)

    if peaks.size == 0:
        # Adaptive fallback for very noisy records.
        threshold = float(np.percentile(integrated, 90))
        peaks, _ = find_peaks(integrated, distance=distance, height=threshold)

    return peaks.astype(np.int64)


def segment_beats_around_r_peaks(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    pre_seconds: float = settings.beat_pre_seconds,
    post_seconds: float = settings.beat_post_seconds,
) -> np.ndarray:
    pre = int(pre_seconds * fs)
    post = int(post_seconds * fs)
    seg_len = pre + post
    if seg_len <= 0:
        return np.empty((0, 0), dtype=np.float32)

    segments: list[np.ndarray] = []
    for peak in r_peaks:
        start = int(peak) - pre
        end = int(peak) + post
        if start < 0 or end > len(signal):
            continue
        seg = signal[start:end]
        if len(seg) == seg_len:
            segments.append(seg.astype(np.float32))
    if not segments:
        return np.empty((0, seg_len), dtype=np.float32)
    return np.stack(segments, axis=0).astype(np.float32)


def fixed_length_segments(
    signal: np.ndarray,
    fs: int,
    segment_seconds: float = settings.segment_seconds,
    overlap: float = 0.5,
) -> np.ndarray:
    seg_len = int(segment_seconds * fs)
    if seg_len <= 0:
        raise ValueError("segment_seconds results in non-positive segment length")
    hop = max(int(seg_len * (1 - overlap)), 1)
    if len(signal) < seg_len:
        padded = np.pad(signal, (0, seg_len - len(signal)), mode="edge")
        return np.expand_dims(padded.astype(np.float32), axis=0)

    segments: list[np.ndarray] = []
    for start in range(0, len(signal) - seg_len + 1, hop):
        segments.append(signal[start : start + seg_len].astype(np.float32))
    return np.stack(segments, axis=0).astype(np.float32)


def signal_snr_db(raw_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    noise = raw_signal - denoised_signal
    signal_power = float(np.mean(denoised_signal**2))
    noise_power = float(np.mean(noise**2))
    if noise_power < 1e-12:
        return 40.0
    return 10.0 * np.log10(max(signal_power, 1e-12) / noise_power)


def classify_signal_quality(raw_signal: np.ndarray, denoised_signal: np.ndarray) -> tuple[int, float]:
    snr_db = signal_snr_db(raw_signal, denoised_signal)
    if snr_db >= settings.quality_good_snr_db:
        return 0, snr_db
    if snr_db >= settings.quality_noisy_snr_db:
        return 1, snr_db
    return 2, snr_db


def preprocess_signal(
    signal: np.ndarray | list[float] | list[list[float]],
    sampling_rate: int,
    target_sampling_rate: int = settings.target_sampling_rate,
) -> PreprocessingResult:
    raw = to_numpy_signal(signal)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("input signal is empty after removing non-finite values")

    resampled = resample_signal(raw, sampling_rate, target_sampling_rate)
    filtered = butter_bandpass_filter(resampled, target_sampling_rate)
    baseline_corrected = remove_baseline_wander(filtered, target_sampling_rate)
    normalized = zscore_normalize(baseline_corrected)

    r_peaks = pan_tompkins_r_peaks(normalized, target_sampling_rate)
    beat_segments = segment_beats_around_r_peaks(normalized, r_peaks, target_sampling_rate)
    fixed_segments = fixed_length_segments(normalized, target_sampling_rate)

    quality_id, snr_db = classify_signal_quality(resampled, baseline_corrected)
    quality_label = quality_label_from_id(quality_id)

    return PreprocessingResult(
        signal=normalized,
        sampling_rate=target_sampling_rate,
        r_peaks=r_peaks,
        beat_segments=beat_segments,
        fixed_segments=fixed_segments,
        quality_id=quality_id,
        quality_label=quality_label,
        snr_db=snr_db,
    )


def summarize_preprocessing(result: PreprocessingResult) -> dict[str, Any]:
    return {
        "sampling_rate": result.sampling_rate,
        "length": int(len(result.signal)),
        "num_r_peaks": int(len(result.r_peaks)),
        "num_beats": int(result.beat_segments.shape[0]),
        "num_fixed_segments": int(result.fixed_segments.shape[0]),
        "quality_id": int(result.quality_id),
        "quality_label": result.quality_label,
        "snr_db": float(result.snr_db),
    }

