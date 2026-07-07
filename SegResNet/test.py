"""Evaluate SegResNet on the held-out test HDF5 dataset."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from monai.metrics import SSIMMetric
from monai.networks.nets import SegResNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.metrics import aggregate_metrics, compute_reconstruction_metrics
from utils.plotting import plot_sample


REQUIRED_STITCH_METADATA = (
    "source_name",
    "source_index",
    "y",
    "x",
    "height",
    "width",
)


class HDF5PatchDataset(Dataset):
    """HDF5 patch dataset that also returns the patch index for stitching."""

    def __init__(self, h5_filename: str) -> None:
        self.h5_filename = h5_filename
        with h5py.File(self.h5_filename, "r") as file:
            self.total_num_samples = len(file["label"])
            missing_metadata = [
                key for key in REQUIRED_STITCH_METADATA if key not in file
            ]
            if missing_metadata:
                raise ValueError(
                    f"{self.h5_filename} lacks stitching metadata: "
                    f"{missing_metadata}. Rebuild it with SegResNet/prepare_data.py."
                )

    def __len__(self) -> int:
        return self.total_num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if not hasattr(self, "opened_hdf5"):
            self.opened_hdf5 = h5py.File(self.h5_filename, "r")
        image = self.opened_hdf5["image"][idx]
        label = self.opened_hdf5["label"][idx]
        return torch.FloatTensor(image), torch.FloatTensor(label), idx

    def __del__(self) -> None:
        if hasattr(self, "opened_hdf5"):
            self.opened_hdf5.close()


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    parser = argparse.ArgumentParser(
        description="Evaluate SegResNet on the test HDF5 dataset."
    )
    parser.add_argument(
        "--test-h5",
        type=Path,
        default=data_dir / "test-4-128.hdf5",
        help="Path to the test HDF5 dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoints_dir / "SegResNet_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for PNG visualizations and metrics.json.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_model(
    checkpoint: Path,
    init_filters: int,
    device: torch.device,
) -> SegResNet:
    model = SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)


def read_stitch_metadata(test_h5: Path) -> Dict[str, np.ndarray]:
    with h5py.File(test_h5, "r") as h5f:
        missing_metadata = [
            key for key in REQUIRED_STITCH_METADATA if key not in h5f
        ]
        if missing_metadata:
            raise ValueError(
                f"{test_h5} lacks stitching metadata: {missing_metadata}. "
                "Rebuild it with SegResNet/prepare_data.py."
            )
        return {
            "source_name": h5f["source_name"].asstr()[:],
            "source_index": h5f["source_index"][:],
            "y": h5f["y"][:],
            "x": h5f["x"][:],
            "height": h5f["height"][:],
            "width": h5f["width"][:],
        }


def make_reconstruction_record(height: int, width: int) -> Dict[str, np.ndarray]:
    shape = (height, width)
    return {
        "input_sum": np.zeros(shape, dtype=np.float64),
        "predict_sum": np.zeros(shape, dtype=np.float64),
        "target_sum": np.zeros(shape, dtype=np.float64),
        "weight": np.zeros(shape, dtype=np.float64),
    }


def add_patch_to_record(
    record: Dict[str, np.ndarray],
    sparse_input: np.ndarray,
    predict: np.ndarray,
    target: np.ndarray,
    y: int,
    x: int,
) -> None:
    patch_height, patch_width = target.shape
    patch_slice = np.s_[y : y + patch_height, x : x + patch_width]
    record["input_sum"][patch_slice] += sparse_input
    record["predict_sum"][patch_slice] += predict
    record["target_sum"][patch_slice] += target
    record["weight"][patch_slice] += 1.0


def finalize_reconstruction(
    record: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight = record["weight"]
    covered = weight > 0
    if not np.any(covered):
        raise ValueError("Cannot stitch a seismogram without covered pixels.")

    sparse_input = np.zeros_like(record["input_sum"], dtype=np.float64)
    predict = np.zeros_like(record["predict_sum"], dtype=np.float64)
    target = np.zeros_like(record["target_sum"], dtype=np.float64)
    sparse_input[covered] = record["input_sum"][covered] / weight[covered]
    predict[covered] = record["predict_sum"][covered] / weight[covered]
    target[covered] = record["target_sum"][covered] / weight[covered]

    covered_rows = np.where(np.any(covered, axis=1))[0]
    covered_cols = np.where(np.any(covered, axis=0))[0]
    y0, y1 = covered_rows[0], covered_rows[-1] + 1
    x0, x1 = covered_cols[0], covered_cols[-1] + 1
    bbox = np.s_[y0:y1, x0:x1]
    return sparse_input[bbox], predict[bbox], target[bbox], covered[bbox]


def build_results_payload(
    args: argparse.Namespace,
    device: torch.device,
    sample_results: List[Dict[str, Any]],
    aggregate: Dict[str, float],
    global_r2: float,
) -> Dict[str, Any]:
    metric_descriptions = {
        "global_r2": (
            "Global Coefficient of Determination over the whole test set: "
            "1 - sum((target - predict)^2) / sum((target - mean(target))^2)."
        ),
        "ssim": (
            "Structural Similarity Index Measure: preservation of wavefield "
            "structure and reflector continuity."
        ),
        "snr": (
            "Signal-to-Noise Ratio: ratio of target signal energy to "
            "reconstruction error energy."
        ),
        "rmse": (
            "Root Mean Squared Error: average quadratic amplitude error."
        ),
        "mae": (
            "Mean Absolute Error: average absolute amplitude error."
        ),
        "residual_correlation": (
            "Pearson correlation between the residual (target - predict) "
            "and the target."
        ),
    }

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint.resolve()),
            "test_h5": str(args.test_h5.resolve()),
            "device": str(device),
            "num_reconstructed_seismograms": len(sample_results),
        },
        "metric_descriptions": metric_descriptions,
        "global_r2": global_r2,
        "aggregate": aggregate,
        "samples": sample_results,
    }


def compute_global_r2(
    ss_res: float,
    target_sum: float,
    target_sq_sum: float,
    target_count: int,
) -> float:
    if target_count == 0:
        raise ValueError("Cannot compute global R2 for an empty test set.")

    ss_tot = target_sq_sum - (target_sum**2) / target_count
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def main() -> None:
    args = parse_args()

    if not args.test_h5.is_file():
        raise FileNotFoundError(f"Test HDF5 not found: {args.test_h5}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    testset = HDF5PatchDataset(str(args.test_h5))
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    model = build_model(args.checkpoint, args.init_filters, device)
    model.eval()
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    stitch_metadata = read_stitch_metadata(args.test_h5)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_results: List[Dict[str, Any]] = []
    sample_metrics: List[Dict[str, float]] = []
    reconstructions: Dict[str, Dict[str, np.ndarray]] = {}
    source_order: List[str] = []

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Evaluating test set"):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            patch_indices = batch[2].cpu().numpy()
            predictions = model(inputs)

            for i in range(inputs.shape[0]):
                patch_idx = int(patch_indices[i])
                source_name = str(stitch_metadata["source_name"][patch_idx])
                if source_name not in reconstructions:
                    source_order.append(source_name)
                    reconstructions[source_name] = make_reconstruction_record(
                        height=int(stitch_metadata["height"][patch_idx]),
                        width=int(stitch_metadata["width"][patch_idx]),
                    )

                sparse_input = inputs[i, 0].cpu().numpy()
                target = targets[i, 0].cpu().numpy()
                predict = predictions[i, 0].cpu().numpy()
                add_patch_to_record(
                    reconstructions[source_name],
                    sparse_input=sparse_input,
                    predict=predict,
                    target=target,
                    y=int(stitch_metadata["y"][patch_idx]),
                    x=int(stitch_metadata["x"][patch_idx]),
                )

    global_ss_res = 0.0
    global_target_sum = 0.0
    global_target_sq_sum = 0.0
    global_target_count = 0

    for sample_idx, source_name in enumerate(
        tqdm(source_order, desc="Scoring stitched seismograms")
    ):
        full_covered_fraction = float(
            np.mean(reconstructions[source_name]["weight"] > 0)
        )
        sparse_input, predict, target, _covered = finalize_reconstruction(
            reconstructions[source_name]
        )

        error = target - predict
        global_ss_res += float(np.sum(error * error))
        global_target_sum += float(np.sum(target))
        global_target_sq_sum += float(np.sum(target * target))
        global_target_count += int(target.size)

        metrics = compute_reconstruction_metrics(
            target=target,
            predict=predict,
            ssim_metric=ssim_metric,
        )
        image_name = f"seismogram_{sample_idx:05d}.png"
        plot_sample(
            sparse_input,
            predict,
            target,
            save_path=args.output_dir / image_name,
        )

        sample_results.append(
            {
                "index": sample_idx,
                "source_name": source_name,
                "image": image_name,
                "covered_fraction": full_covered_fraction,
                "metrics": metrics,
            }
        )
        sample_metrics.append(metrics)

    aggregate = aggregate_metrics(sample_metrics)
    global_r2 = compute_global_r2(
        ss_res=global_ss_res,
        target_sum=global_target_sum,
        target_sq_sum=global_target_sq_sum,
        target_count=global_target_count,
    )
    results_payload = build_results_payload(
        args=args,
        device=device,
        sample_results=sample_results,
        aggregate=aggregate,
        global_r2=global_r2,
    )

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results_payload, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(sample_results)} visualizations to {args.output_dir}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Global R2: {global_r2:.6f}")
    print("Aggregate metrics:")
    for name, value in aggregate.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()

