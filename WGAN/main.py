import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import generator_dense, critic_dense
from wgan import WGAN
from utils.data_utils import LoadImageDataset
from utils.plotting_functions import plotting_image_grid
from config import cla


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

    # train_data = LoadImageDataset(
    #     datafile=PARAMS.train_file, N=PARAMS.n_train, permute=True, x_C=1
    # )

    # loader = DataLoader(
    #     train_data, batch_size=PARAMS.batch_size, shuffle=True, drop_last=True
    # )

    # Creating the models

    generator_model = generator_dense(z_dim=50)
    critic_model = critic_dense()
    generator_model.to(device)
    critic_model.to(device)

    summary(generator_model, input_size=(50,))
    summary(critic_model, input_size=(1, 64, 64))

    g_optim = Adam(
        generator_model.parameters(),
        lr=PARAMS.learn_rate,
        betas=(0.5, 0.9),
        amsgrad=True,
    )
    c_optim = Adam(
        critic_model.parameters(),
        lr=PARAMS.learn_rate,
        betas=(0.5, 0.9),
        amsgrad=True,
    )

    wgan_trainer = WGAN(
        directory=PARAMS.saving_dir,
        device=device,
        batch_size=PARAMS.batch_size,
        n_critic=PARAMS.n_critic,
        gp_coef=PARAMS.gp_coef,
        z_dim=PARAMS.z_dim,
        learn_rate=PARAMS.learn_rate,
    )

    wgan_trainer.load_models_init(generator_model, critic_model, g_optim, c_optim)

    wgan_trainer.train(
        train_data=loader, n_epoch=4000, sample_plotter=plotting_image_grid
    )


if __name__ == "__main__":
    main()
