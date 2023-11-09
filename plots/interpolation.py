import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import torch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def get_perplexity(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    # iterate over the lines from the end
    for line in reversed(lines):
        if "evaluation perplexity:" in line:
            _, perplexity = line.split("evaluation perplexity:")
            return float(perplexity)

    return None


if __name__ == "__main__":
    kernel_size = 40
    min_loss = 14
    max_scaler = 1
    log_level = 1  # + len(modules)

    fig = plt.figure(figsize=(6 * 3, 5 * 3))  # , layout='tight')
    gs = gridspec.GridSpec(3, 3)

    exp_dir = "/fsx/home-mitchellw/experimetns/lm/"

    ax = fig.add_subplot(gs[0, 0])

    for j, base in enumerate(
        [
            #'/fsx/home-mitchellw/experimetns/lmtune/instruction-tune-1b-2e-5-6',
            "/fsx/home-mitchellw/experimetns/lmtune/instruction-tune-3b-2e-5-6",
        ]
    ):
        xs, ys, colors = [], [], []
        for alpha in np.arange(0, 1.01, 0.05):
            chat_eval = f"{base}/checkpoints/chat-eval-interpolate-{alpha:.2f}-epoch_6.pt"
            base_eval = f"{base}/checkpoints/base-eval-interpolate-{alpha:.2f}-epoch_6.pt"
            if os.path.exists(chat_eval) and os.path.exists(base_eval):
                chat_y = get_perplexity(chat_eval)
                base_y = get_perplexity(base_eval)
                if chat_y is None or base_y is None:
                    continue
                print(alpha)
                xs.append(base_y)
                ys.append(chat_y)
                colors.append(1 - alpha)  # add alpha to the color list

        scatter = ax.scatter(
            xs,
            ys,
            c=colors,
            cmap="cool",
            marker="d" if "3B" in base else "o",
            label="OpenLM-1B" if "3B" in base else "OpenLM-3B",
        )

    ax.set_xlabel("Base evaluation set (perplexity)", fontsize=12)
    ax.set_ylabel("Chat evaluation set (perplexity)", fontsize=12)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid()

    ax.legend(fontsize=12)

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(
        "Interpolation coefficient when interpolating\nbetween base and chat models",
        labelpad=10,
    )

    plt.savefig("plots/interpolation.png", bbox_inches="tight")
