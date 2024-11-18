import torch.nn as nn
from DL_toolbox.models.architecture_blocks import UpSample2D, DownSample2D, DenseBlock2D


class generator_dense(nn.Module):
    """Generator model"""

    def __init__(self, z_dim=50, cnn_init_dim=(512, 4, 4), act_param=0.1):
        """
        x_shape does not include the number of samples N.
        """
        super(generator_dense, self).__init__()

        self.units_init = cnn_init_dim[0] * cnn_init_dim[1] * cnn_init_dim[2]
        self.z_dim = z_dim
        self.cnn_init_dim = cnn_init_dim
        self.act_param = act_param

        self.l1 = nn.Linear(self.z_dim, self.units_init)
        self.l2 = UpSample2D(
            x_dim=cnn_init_dim[0],
            filters=cnn_init_dim[0] // 2,
            act_param=self.act_param,
        )
        self.l3 = DenseBlock2D(
            x_shape=(cnn_init_dim[0] // 2, cnn_init_dim[1] * 2, cnn_init_dim[2] * 2),
            normalization="bn",
            act_param=self.act_param,
            out_channels=cnn_init_dim[0] // 2,
            layers=2,
        )
        self.l4 = UpSample2D(
            x_dim=cnn_init_dim[0] // 2,
            filters=cnn_init_dim[0] // 4,
            act_param=self.act_param,
        )
        self.l5 = DenseBlock2D(
            x_shape=(cnn_init_dim[0] // 4, cnn_init_dim[1] * 4, cnn_init_dim[2] * 4),
            normalization="bn",
            act_param=self.act_param,
            out_channels=cnn_init_dim[0] // 4,
            layers=3,
        )
        self.l6 = UpSample2D(
            x_dim=cnn_init_dim[0] // 4,
            filters=cnn_init_dim[0] // 8,
            act_param=self.act_param,
        )
        self.l7 = DenseBlock2D(
            x_shape=(cnn_init_dim[0] // 8, cnn_init_dim[1] * 8, cnn_init_dim[2] * 8),
            normalization="bn",
            act_param=self.act_param,
            out_channels=cnn_init_dim[0] // 8,
            layers=3,
        )
        self.l8 = UpSample2D(
            x_dim=cnn_init_dim[0] // 8,
            filters=cnn_init_dim[0] // 16,
            act_param=self.act_param,
        )
        self.l9 = DenseBlock2D(
            x_shape=(cnn_init_dim[0] // 16, cnn_init_dim[1] * 16, cnn_init_dim[2] * 16),
            normalization="bn",
            act_param=self.act_param,
            out_channels=cnn_init_dim[0] // 16,
            layers=3,
        )
        self.l10 = UpSample2D(
            x_dim=cnn_init_dim[0] // 16,
            filters=1,
            upsample=False,
            act_param=self.act_param,
        )

        self.ELU = nn.ELU(alpha=act_param)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_z):
        x1 = self.l1(input_z)
        x1 = self.ELU(x1)
        x2 = x1.view(
            size=(-1, self.cnn_init_dim[0], self.cnn_init_dim[1], self.cnn_init_dim[2])
        )
        x2 = self.l2(x2)

        x2 = self.l3(x2)

        x3 = self.l4(x2)
        x3 = self.l5(x3)

        x4 = self.l6(x3)
        x4 = self.l7(x4)

        x5 = self.l8(x4)
        x5 = self.l9(x5)

        x5 = self.l10(x5)

        output = x5

        return output


class critic_dense(nn.Module):
    """Critic model: U-Net with skip connections and Resblocks
    input_x is assumed to have the shape (N, C, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    """

    def __init__(self, init_dim=(1, 64, 64), init_k=16, act_param=0.1):
        """
        x_shape does not include the number of samples N.
        """
        super(critic_dense, self).__init__()

        self.init_dim = init_dim
        self.act_param = act_param
        self.init_k = init_k

        self.l1 = UpSample2D(
            x_dim=self.init_dim[0],
            filters=self.init_k,
            upsample=False,
            act_param=self.act_param,
        )
        self.l2 = DenseBlock2D(
            x_shape=(self.init_k, self.init_dim[1], self.init_dim[2]),
            normalization="ln",
            act_param=self.act_param,
            out_channels=self.init_k,
            layers=3,
        )
        self.l3 = DownSample2D(
            x_dim=self.init_k,
            filters=self.init_k * 2,
            act_param=self.act_param,
        )
        self.l4 = DenseBlock2D(
            x_shape=(self.init_k * 2, self.init_dim[1] // 2, self.init_dim[2] // 2),
            normalization="ln",
            act_param=self.act_param,
            out_channels=self.init_k * 2,
            layers=3,
        )
        self.l5 = DownSample2D(
            x_dim=self.init_k * 2,
            filters=self.init_k * 4,
            act_param=self.act_param,
        )
        self.l6 = DenseBlock2D(
            x_shape=(self.init_k * 4, self.init_dim[1] // 4, self.init_dim[2] // 4),
            normalization="ln",
            act_param=self.act_param,
            out_channels=self.init_k * 4,
            layers=3,
        )
        self.l7 = DownSample2D(
            x_dim=self.init_k * 4,
            filters=self.init_k * 8,
            act_param=self.act_param,
        )
        self.l8 = DenseBlock2D(
            x_shape=(self.init_k * 8, self.init_dim[1] // 8, self.init_dim[2] // 8),
            normalization="ln",
            act_param=self.act_param,
            out_channels=self.init_k * 8,
            layers=3,
        )
        self.l9 = DownSample2D(
            x_dim=self.init_k * 8,
            filters=self.init_k * 16,
            act_param=self.act_param,
        )
        self.l10 = DenseBlock2D(
            x_shape=(self.init_k * 16, self.init_dim[1] // 16, self.init_dim[2] // 16),
            normalization="ln",
            act_param=self.act_param,
            out_channels=self.init_k * 16,
            layers=2,
        )
        self.l11 = nn.Linear(
            self.init_k * self.init_dim[1] * self.init_dim[2] // 16, 128
        )
        self.l12 = nn.Linear(128, 64)
        self.l13 = nn.Linear(64, 1)

        self.flatten = nn.Flatten()
        self.LReLU = nn.ELU(alpha=self.act_param)

    def forward(self, input_data):
        x1 = self.l1(input_data)
        x1 = self.l2(x1)

        x2 = self.l3(x1)
        x2 = self.l4(x2)

        x3 = self.l5(x2)
        x3 = self.l6(x3)

        x4 = self.l7(x3)
        x4 = self.l8(x4)

        x5 = self.l9(x4)
        x5 = self.l10(x5)

        x6 = self.flatten(x5)

        x7 = self.l11(x6)
        x7 = self.LReLU(x7)

        x8 = self.l12(x7)
        x8 = self.LReLU(x8)

        output = self.l13(x8)

        return output
