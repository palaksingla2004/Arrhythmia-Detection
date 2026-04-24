from __future__ import annotations

import random

import numpy as np
from scipy.signal import resample


class ECGAugmenter:
    def __init__(
        self,
        noise_std: float = 0.02,
        max_shift_ratio: float = 0.1,
        time_scale_range: tuple[float, float] = (0.9, 1.1),
        crop_ratio_range: tuple[float, float] = (0.85, 1.0),
        p_noise: float = 0.6,
        p_shift: float = 0.5,
        p_scale: float = 0.5,
        p_crop: float = 0.5,
    ) -> None:
        self.noise_std = noise_std
        self.max_shift_ratio = max_shift_ratio
        self.time_scale_range = time_scale_range
        self.crop_ratio_range = crop_ratio_range
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.p_scale = p_scale
        self.p_crop = p_crop

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        x = signal.astype(np.float32).copy()
        target_len = len(x)

        if random.random() < self.p_noise:
            x = self._add_gaussian_noise(x)
        if random.random() < self.p_shift:
            x = self._shift_signal(x)
        if random.random() < self.p_scale:
            x = self._time_scale(x, target_len)
        if random.random() < self.p_crop:
            x = self._random_crop_resize(x, target_len)
        return x.astype(np.float32)

    def _add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=len(x)).astype(np.float32)
        return x + noise

    def _shift_signal(self, x: np.ndarray) -> np.ndarray:
        max_shift = int(len(x) * self.max_shift_ratio)
        if max_shift <= 1:
            return x
        shift = random.randint(-max_shift, max_shift)
        return np.roll(x, shift)

    def _time_scale(self, x: np.ndarray, target_len: int) -> np.ndarray:
        scale = random.uniform(self.time_scale_range[0], self.time_scale_range[1])
        scaled_len = max(int(target_len * scale), 16)
        scaled = resample(x, scaled_len).astype(np.float32)
        return self._pad_or_crop(scaled, target_len)

    def _random_crop_resize(self, x: np.ndarray, target_len: int) -> np.ndarray:
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_len = max(int(target_len * crop_ratio), 16)
        if crop_len >= len(x):
            return x
        start = random.randint(0, len(x) - crop_len)
        cropped = x[start : start + crop_len]
        return resample(cropped, target_len).astype(np.float32)

    @staticmethod
    def _pad_or_crop(x: np.ndarray, target_len: int) -> np.ndarray:
        if len(x) == target_len:
            return x
        if len(x) > target_len:
            start = (len(x) - target_len) // 2
            return x[start : start + target_len]
        pad_left = (target_len - len(x)) // 2
        pad_right = target_len - len(x) - pad_left
        return np.pad(x, (pad_left, pad_right), mode="edge")

