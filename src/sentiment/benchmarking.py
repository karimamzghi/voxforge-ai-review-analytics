"""Small inference-speed benchmark utility."""
from __future__ import annotations

import platform
import time
from pathlib import Path
import pandas as pd
import torch


def benchmark_predictor(predictor, texts, *, model_name: str, batch_sizes=(1, 8, 32), timed_runs: int = 3) -> pd.DataFrame:
    values = list(texts)
    if not values:
        raise ValueError("At least one text is required.")
    rows = []
    for batch_size in batch_sizes:
        sample = values[: min(len(values), max(batch_size * 10, 100))]
        predictor.batch_size = batch_size
        predictor.predict_batch(sample[: min(batch_size, len(sample))])  # warm-up
        durations = []
        for _ in range(timed_runs):
            start = time.perf_counter(); predictor.predict_batch(sample); durations.append(time.perf_counter() - start)
        average = sum(durations) / len(durations)
        rows.append({
            "model_name": model_name,
            "device": predictor.device,
            "batch_size": batch_size,
            "sample_count": len(sample),
            "timed_runs": timed_runs,
            "total_seconds": average,
            "reviews_per_second": len(sample) / average,
            "average_ms_per_review": average / len(sample) * 1000,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        })
    return pd.DataFrame(rows)


def save_speed_benchmark(frames, output_path: str | Path) -> Path:
    result = pd.concat(list(frames), ignore_index=True)
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return path
