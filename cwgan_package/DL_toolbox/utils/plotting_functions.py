import matplotlib.pyplot as plt
import numpy as np


def plotting_image_grid(images, savedir, grid=(4, 4)):
    fig, ax = plt.subplots(grid[0], grid[1], figsize=(grid[0] * 3, grid[1] * 3))

    for i in range(grid[0] * grid[1]):
        ax[i // 4, i % 4].imshow(images[i, 0, :, :])
        ax[i // 4, i % 4].set_xticks([])
        ax[i // 4, i % 4].set_yticks([])

    fig.subplots_adjust(
        wspace=0.04, hspace=0.04, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    plt.savefig(savedir)
    plt.close()


def plotting_image_grid_cwgan_2to1(
    predictions, true_y, true_x, n_stat, savedir, grid=(4, 5)
):
    fig, ax = plt.subplots(grid[0], grid[1], figsize=(grid[0] * 3, grid[1] * 3))

    for i in range(grid[0]):
        ax[i, 0].imshow(true_y[i, 0, :, :])
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        ax[i, 1].imshow(true_y[i, 1, :, :])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])

        ax[i, 2].imshow(true_x[i, 0, :, :], vmax=1)
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])

        ax[i, 3].imshow(
            np.mean(predictions[i * n_stat : (i + 1) * n_stat, 0, :, :], axis=0),
            vmax=1,
        )
        ax[i, 3].set_xticks([])
        ax[i, 3].set_yticks([])

        ax[i, 4].imshow(
            np.std(predictions[i * n_stat : (i + 1) * n_stat, 0, :, :], axis=0),
            vmax=1,
        )
        ax[i, 4].set_xticks([])
        ax[i, 4].set_yticks([])

    fig.subplots_adjust(
        wspace=0.04, hspace=0.04, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    plt.savefig(savedir)
    plt.close()
