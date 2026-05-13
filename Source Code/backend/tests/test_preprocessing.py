import numpy as np

from app.data.preprocessing import preprocess_signal


def test_preprocess_signal_runs():
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    signal = 0.1 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.01, size=len(t))
    result = preprocess_signal(signal, sampling_rate=fs)
    assert result.signal.ndim == 1
    assert result.fixed_segments.shape[1] > 0
    assert result.quality_label in {"Good", "Noisy", "Unusable"}

