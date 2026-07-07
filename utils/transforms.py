import copy

import albumentations as A
import numpy as np


class MinMaxNormalize(A.ImageOnlyTransform):
    """MinMax normalization to (0, 1) for A.Compose pipelines."""

    def __init__(
        self,
        eps: float = 1e-7,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.eps = eps

    def apply(self, img, **params):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + self.eps)


class MakeSparce(A.ImageOnlyTransform):
    """Sparsify input by zeroing regular or random traces/columns."""

    def __init__(
        self,
        method: str = "regular",
        regular_step: int = 2,
        random_keep_rate: float = 0.7,
        axis_x: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.axis_x = axis_x
        self.step = regular_step
        self.keep_rate = random_keep_rate
        self.method = method

    def apply(self, img, **params):
        mask = np.zeros(img.shape, dtype=bool)
        if self.method == "regular":
            if self.axis_x:
                mask[:, :: self.step] = True
            else:
                mask[:: self.step, :] = True
        elif self.method == "random":
            if self.axis_x:
                mask[:, np.random.random(img.shape[1]) < 1 - self.keep_rate] = True
            else:
                mask[np.random.random(img.shape[0]) < 1 - self.keep_rate, :] = True
        else:
            raise ValueError("method must be 'regular' or 'random'")

        result = copy.deepcopy(img)
        result[mask] = 0.0
        return result


def build_inputs_pipeline(
    random_keep_rate: float = 0.85,
    regular_step: int = 2,
) -> A.Compose:
    return A.Compose(
        [
            MinMaxNormalize(always_apply=True, p=1.0),
            MakeSparce(
                method="random",
                random_keep_rate=random_keep_rate,
                regular_step=regular_step,
                always_apply=True,
                p=1.0,
            ),
        ]
    )


def build_targets_pipeline() -> A.Compose:
    return A.Compose([MinMaxNormalize(always_apply=True, p=1.0)])

