"""Prepare train/val/test HDF5 datasets from raw .npy files (MakeHDF5Dataset.ipynb)."""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import h5py
import numpy as np
from tqdm import tqdm 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.tiling import (
    crop_to_samples,
    get_tile_coords,
    get_tiles_num,
)
from utils.transforms import build_inputs_pipeline, build_targets_pipeline

DEFAULT_RAW_DATA = "data/npy_data_cut_156"


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(
        description="Build HDF5 train/val/test datasets."
    )
    parser.add_argument(
        "--raw-data-folder",
        type=Path,
        default=Path(DEFAULT_RAW_DATA),
        help="Folder with source .npy files.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=data_dir / "train-4-128.hdf5",
        help="Output path for train HDF5.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=data_dir / "val-4-128.hdf5",
        help="Output path for validation HDF5.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=data_dir / "test-4-128.hdf5",
        help="Output path for test HDF5.",
    )
    parser.add_argument(
        "--skip-size-coef",
        type=int,
        default=1,
        help="Use every Nth .npy file. The default 1 uses all files.",
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument(
        "--crop-stride-coef",
        type=float,
        default=0.5,
        help=(
            "Patch stride as a fraction of tile size. Values below 1.0 create "
            "overlap; the default 0.5 means 50%% overlap."
        ),
    )
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-keep-rate", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 files instead of requiring compatible shapes.",
    )
    return parser.parse_args()


def split_filenames(
    filenames: List[str],
    train_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be in (0, 1)")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be in (0, 1)")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1")

    rng = np.random.default_rng(seed)
    shuffled_idxs = rng.permutation(len(filenames))
    train_count = int(len(filenames) * train_size)
    val_count = int(len(filenames) * val_size)

    train_files = [filenames[idx] for idx in shuffled_idxs[:train_count]]
    val_files = [
        filenames[idx]
        for idx in shuffled_idxs[train_count : train_count + val_count]
    ]
    test_files = [filenames[idx] for idx in shuffled_idxs[train_count + val_count :]]
    return train_files, val_files, test_files


def count_tiles(
    raw_data_folder: Path,
    filenames: Iterable[str],
    tile_size: int,
    stride: float,
) -> int:
    total_tiles = 0
    for filename in filenames:
        data_sample = np.load(raw_data_folder / filename)
        total_tiles += get_tiles_num(
            data_sample,
            tile_size=tile_size,
            stride=stride,
        )
    return total_tiles


def fill_hdf5_dataset(
    dataset: h5py.File,
    filenames: Iterable[str],
    raw_data_folder: Path,
    inputs_pipeline,
    targets_pipeline,
    tile_size: int,
    stride: float,
    desc: str,
) -> None:
    current_idx = 0
    for source_idx, filename in enumerate(tqdm(list(filenames), desc=desc)):
        data_sample = np.load(raw_data_folder / filename)
        inputs = inputs_pipeline(image=data_sample)["image"]
        targets = targets_pipeline(image=data_sample)["image"]
        inputs_cut, targets_cut = crop_to_samples(
            inputs,
            targets=targets,
            tile_size=tile_size,
            stride=stride,
        )
        coords = get_tile_coords(
            targets,
            tile_size=tile_size,
            stride=stride,
        )

        for i in range(len(inputs_cut)):
            dataset["image"][current_idx] = copy.deepcopy(inputs_cut[i])
            dataset["label"][current_idx] = copy.deepcopy(targets_cut[i])
            dataset["source_name"][current_idx] = filename
            dataset["source_index"][current_idx] = source_idx
            dataset["y"][current_idx] = coords[i, 0]
            dataset["x"][current_idx] = coords[i, 1]
            dataset["height"][current_idx] = targets.shape[0]
            dataset["width"][current_idx] = targets.shape[1]
            current_idx += 1


def create_hdf5_datasets(
    datasets: Iterable[Tuple[Path, int]],
    tile_size: int,
    overwrite: bool = False,
) -> None:
    expected_shape = (1, tile_size, tile_size)
    for path, count in datasets:
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                with h5py.File(path, "r") as h5f:
                    image_shape = h5f["image"].shape
                    label_shape = h5f["label"].shape
                    required_metadata = (
                        "source_name",
                        "source_index",
                        "y",
                        "x",
                        "height",
                        "width",
                    )
                    missing_metadata = [
                        name for name in required_metadata if name not in h5f
                    ]
                if (
                    image_shape != (count, *expected_shape)
                    or label_shape != (count, *expected_shape)
                    or missing_metadata
                ):
                    raise ValueError(
                        f"Existing HDF5 has incompatible shape: {path}. "
                        f"Expected image/label shape {(count, *expected_shape)}, "
                        f"got image {image_shape}, label {label_shape}. "
                        "Choose a different output path, remove the old file, "
                        "or rerun with --overwrite."
                    )
                continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5f:
            h5f.create_dataset(
                "image",
                shape=(count, 1, tile_size, tile_size),
                dtype=np.float32,
            )
            h5f.create_dataset(
                "label",
                shape=(count, 1, tile_size, tile_size),
                dtype=np.float32,
            )
            h5f.create_dataset(
                "source_name",
                shape=(count,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5f.create_dataset("source_index", shape=(count,), dtype=np.int32)
            h5f.create_dataset("y", shape=(count,), dtype=np.int32)
            h5f.create_dataset("x", shape=(count,), dtype=np.int32)
            h5f.create_dataset("height", shape=(count,), dtype=np.int32)
            h5f.create_dataset("width", shape=(count,), dtype=np.int32)


def main() -> None:
    args = parse_args()
    raw_data_folder = args.raw_data_folder

    if not raw_data_folder.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {raw_data_folder}")

    inputs_pipeline = build_inputs_pipeline(random_keep_rate=args.random_keep_rate)
    targets_pipeline = build_targets_pipeline()

    if args.skip_size_coef < 1:
        raise ValueError("skip_size_coef must be >= 1")
    if args.crop_stride_coef <= 0:
        raise ValueError("crop_stride_coef must be > 0")

    all_npy_files = [
        name
        for name in np.sort(os.listdir(raw_data_folder))
        if name.endswith(".npy")
    ]
    npy_files = all_npy_files[:: args.skip_size_coef]
    print(
        f"Found .npy files: {len(all_npy_files)}; "
        f"selected: {len(npy_files)} (skip_size_coef={args.skip_size_coef})"
    )

    train_files, val_files, test_files = split_filenames(
        npy_files,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(
        f"Train files: {len(train_files)}, "
        f"Val files: {len(val_files)}, "
        f"Test files: {len(test_files)}"
    )

    train_tiles_num = count_tiles(
        raw_data_folder,
        train_files,
        tile_size=args.tile_size,
        stride=args.crop_stride_coef,
    )
    val_tiles_num = count_tiles(
        raw_data_folder,
        val_files,
        tile_size=args.tile_size,
        stride=args.crop_stride_coef,
    )
    test_tiles_num = count_tiles(
        raw_data_folder,
        test_files,
        tile_size=args.tile_size,
        stride=args.crop_stride_coef,
    )
    print(
        f"Train tiles: {train_tiles_num}, "
        f"Val tiles: {val_tiles_num}, "
        f"Test tiles: {test_tiles_num}"
    )

    create_hdf5_datasets(
        (
            (args.train_output, train_tiles_num),
            (args.val_output, val_tiles_num),
            (args.test_output, test_tiles_num),
        ),
        args.tile_size,
        overwrite=args.overwrite,
    )

    train_dataset = h5py.File(args.train_output, "a")
    val_dataset = h5py.File(args.val_output, "a")
    test_dataset = h5py.File(args.test_output, "a")

    fill_hdf5_dataset(
        train_dataset,
        train_files,
        raw_data_folder=raw_data_folder,
        inputs_pipeline=inputs_pipeline,
        targets_pipeline=targets_pipeline,
        tile_size=args.tile_size,
        stride=args.crop_stride_coef,
        desc="Building train HDF5",
    )
    fill_hdf5_dataset(
        val_dataset,
        val_files,
        raw_data_folder=raw_data_folder,
        inputs_pipeline=inputs_pipeline,
        targets_pipeline=targets_pipeline,
        tile_size=args.tile_size,
        stride=args.crop_stride_coef,
        desc="Building val HDF5",
    )
    fill_hdf5_dataset(
        test_dataset,
        test_files,
        raw_data_folder=raw_data_folder,
        inputs_pipeline=inputs_pipeline,
        targets_pipeline=targets_pipeline,
        tile_size=args.tile_size,
        stride=args.crop_stride_coef,
        desc="Building test HDF5",
    )

    train_dataset.close()
    val_dataset.close()
    test_dataset.close()
    print(f"Saved train dataset to {args.train_output}")
    print(f"Saved val dataset to {args.val_output}")
    print(f"Saved test dataset to {args.test_output}")


if __name__ == "__main__":
    main()
    
