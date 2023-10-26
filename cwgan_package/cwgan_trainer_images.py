import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from DL_toolbox.models.cwgan import generator_dense_2x64x64_to_1x64x64
from DL_toolbox.models.cwgan import critic_dense_2x64x76_to_1x64x64
from DL_toolbox.methods.cwgan import Conditional_WGAN
from DL_toolbox.utils.data_utils import LoadImageDataset
from DL_toolbox.utils.plotting_functions import plotting_image_grid_cwgan_2to1
from cwgan_config import cla


def main():
    if torch.cuda.is_available():
        print("GPU available")
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    PARAMS = cla()

    np.random.seed(PARAMS.seed_no)
    torch.manual_seed(PARAMS.seed_no)

    train_data = LoadImageDataset(
        datafile=PARAMS.train_file,
        N=PARAMS.n_train,
        divide=True,
        x_C=1,
        y_C=2,
        permute=True,
    )

    loader = DataLoader(
        train_data, batch_size=PARAMS.batch_size, shuffle=True, drop_last=True
    )

    # Creating the models

    x_shape = (1, 64, 64)
    y_shape = (2, 64, 64)
    z_shape = (50, 1, 1)

    generator_model = generator_dense_2x64x64_to_1x64x64(
        y_shape, out_channels=1, z_dim=50
    )
    critic_model = critic_dense_2x64x76_to_1x64x64(x_shape, y_shape[0])

    # summary(
    #     generator_model,
    #     [
    #         y_shape,
    #         z_shape,
    #     ],
    # )
    # summary(critic_model, input_size=[(1, 64, 64), (2, 64, 64)])

    g_optim = Adam(
        generator_model.parameters(),
        lr=PARAMS.learn_rate,
        betas=(0.5, 0.9),
    )
    c_optim = Adam(
        critic_model.parameters(),
        lr=PARAMS.learn_rate,
        betas=(0.5, 0.9),
    )

    wgan_trainer = Conditional_WGAN(
        directory=PARAMS.saving_dir, device=device, z_dim=50, gp_coef=10
    )

    wgan_trainer.load_models_init(generator_model, critic_model, g_optim, c_optim)

    wgan_trainer.train(
        train_data=loader, n_epoch=4000, sample_plotter=plotting_image_grid_cwgan_2to1
    )


if __name__ == "__main__":
    main()
