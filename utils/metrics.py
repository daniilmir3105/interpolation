from typing import Dict, Iterable, List

import numpy as np
import torch
from monai.metrics import SSIMMetric


def compute_reconstruction_metrics(
    target: np.ndarray,
    predict: np.ndarray,
    ssim_metric: SSIMMetric,
) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    predict = np.asarray(predict, dtype=np.float64)
    error = target - predict

    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    signal_energy = float(np.sum(target**2))
    noise_energy = float(np.sum(error**2))
    if noise_energy < 1e-12:
        snr = float("inf")
    else:
        snr = float(10.0 * np.log10(signal_energy / noise_energy))

    target_flat = target.ravel()
    residual_flat = error.ravel()
    if (
        np.std(residual_flat) < 1e-12
        or np.std(target_flat) < 1e-12
    ):
        residual_correlation = 0.0
    else:
        residual_correlation = float(
            np.corrcoef(residual_flat, target_flat)[0, 1]
        )

    target_tensor = torch.as_tensor(
        target[np.newaxis, np.newaxis, ...],
        dtype=torch.float32,
    )
    predict_tensor = torch.as_tensor(
        predict[np.newaxis, np.newaxis, ...],
        dtype=torch.float32,
    )
    ssim = float(ssim_metric(predict_tensor, target_tensor).item())

    return {
        "ssim": ssim,
        "snr": snr,
        "rmse": rmse,
        "mae": mae,
        "residual_correlation": residual_correlation,
    }


def aggregate_metrics(sample_metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics_list: List[Dict[str, float]] = list(sample_metrics)
    if not metrics_list:
        raise ValueError("Cannot aggregate metrics from an empty collection.")

    keys = metrics_list[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [sample[key] for sample in metrics_list if np.isfinite(sample[key])]
        aggregated[key] = float(np.mean(values)) if values else float("nan")
    return aggregated
