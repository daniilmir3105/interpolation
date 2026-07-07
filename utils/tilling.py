import copy
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def get_tiles_num(
    mask: np.ndarray,
    tile_size: int = 512,
    stride: float = 1,
) -> int:
    height, width = mask.shape
    step = int(tile_size * stride)
    return ((height - tile_size + 1) // step + 1) * (
        (width - tile_size + 1) // step + 1
    )


def get_tile_coords(
    mask: np.ndarray,
    tile_size: int = 512,
    stride: float = 1,
) -> npt.NDArray[np.int32]:
    height, width = mask.shape
    step = int(tile_size * stride)
    coords = [
        (y, x)
        for y in range(0, height - tile_size + 1, step)
        for x in range(0, width - tile_size + 1, step)
    ]
    return np.asarray(coords, dtype=np.int32)


def get_idxs_train_test_split(
    total_num: int,
    train_size: float = 0.75,
    seed: Optional[int] = None,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    rng = np.random.default_rng(seed)
    total_range = np.arange(total_num, dtype=np.int32)
    train_count = int(total_num * train_size)
    train_idxs = rng.choice(total_range, train_count, replace=False)
    test_idxs = np.setdiff1d(total_range, train_idxs)
    return train_idxs, test_idxs


def get_idxs_train_val_test_split(
    total_num: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be in (0, 1)")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be in (0, 1)")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.arange(total_num, dtype=np.int32))

    train_count = int(total_num * train_size)
    val_count = int(total_num * val_size)
    train_idxs = shuffled[:train_count]
    val_idxs = shuffled[train_count : train_count + val_count]
    test_idxs = shuffled[train_count + val_count :]
    return train_idxs, val_idxs, test_idxs


def crop_to_samples(
    inputs: np.ndarray,
    targets: np.ndarray,
    tile_size: int = 512,
    stride: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = targets.shape
    step = int(tile_size * stride)
    tiles_num = get_tiles_num(targets, tile_size=tile_size, stride=stride)

    augmented_data = np.zeros(
        (tiles_num, 1, tile_size, tile_size), dtype=np.float32
    )
    augmented_mask = np.zeros(
        (tiles_num, 1, tile_size, tile_size), dtype=np.float32
    )

    t_num = 0
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            augmented_data[t_num] = copy.deepcopy(
                inputs[np.newaxis, y : y + tile_size, x : x + tile_size]
            )
            augmented_mask[t_num] = copy.deepcopy(
                targets[np.newaxis, y : y + tile_size, x : x + tile_size]
            )
            t_num += 1

    return augmented_data, augmented_mask
