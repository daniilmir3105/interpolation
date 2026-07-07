import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uuid
from pathlib import Path
from typing import Optional

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import numpy as np


def plot_sample(
    sample: np.ndarray,
    predict: np.ndarray,
    label: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    fig = plt.figure(figsize=(12, 8), dpi=250)
    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(sample, cmap="gray", vmin=0, vmax=1, aspect="auto")
    plt.axis(False)
    plt.subplot(1, 4, 2)
    plt.title("Predict")
    plt.imshow(predict, cmap="gray", vmin=0, vmax=1, aspect="auto")
    plt.axis(False)
    plt.subplot(1, 4, 3)
    plt.title("Target")
    plt.imshow(label, cmap="gray", vmin=0, vmax=1, aspect="auto")
    plt.axis(False)
    plt.subplot(1, 4, 4)
    plt.title(f"P-T, diff: {np.max(np.abs(predict - label)):.2f}")
    plt.imshow(np.abs(predict - label), cmap="gray", vmin=0, vmax=1, aspect="auto")
    plt.axis(False)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    elif show:
        plt.show()
    else:
        fig.savefig(str(uuid.uuid4()) + ".png", bbox_inches="tight")

    plt.close(fig)
