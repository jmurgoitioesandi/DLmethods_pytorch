import torch.nn as nn
from DL_toolbox.models.architecture_blocks import (
    UpSample,
    DownSample,
    DenseBlock,
    ApplyNormalization,
)
from torch import cat


class generator_dense_2x64x64_to_1x64x64(nn.Module):
    """Generator model: U-Net with skip connections and Denseblocks
    input_x is assumed to have the shape (N, C, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    """

    def __init__(
        self,
        y_shape,
        out_channels=1,
        z_dim=50,
        k0=20,
        act_param=0.1,
        denselayers=3,
        dense_int_out=16,
        g_out="x",
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(generator_dense_2x64x64_to_1x64x64, self).__init__()
        C0, H0, W0 = y_shape

        if z_dim == None:
            normalization = "in"
        else:
            normalization = "cin"

        # ------ Down branch -----------------------------
        H, W, k = H0, W0, k0
        self.d1 = DownSample(x_dim=C0, filters=k, downsample=False, act_param=act_param)
        self.d2 = DenseBlock(
            x_shape=(k, H, W),
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.d3 = DownSample(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, 2 * k
        self.d4 = DenseBlock(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.d5 = DownSample(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, 2 * k
        self.d6 = DenseBlock(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.d7 = DownSample(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, 2 * k
        self.d8 = DenseBlock(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )
        # ------------------------------------------------

        # ----- Base of UNet------------------------------
        self.base = DenseBlock(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )
        # -------------------------------------------------

        # ------ Up branch -----------------------------
        self.u1 = UpSample(x_dim=k, filters=k, act_param=act_param)
        H, W, k = 2 * H, 2 * W, k // 2
        self.u2 = DenseBlock(
            x_shape=(2 * k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.u3 = UpSample(
            x_dim=2 * k, filters=k, concat=True, old_x_dim=k, act_param=act_param
        )
        H, W, k = 2 * H, 2 * W, k // 2
        self.u4 = DenseBlock(
            x_shape=(2 * k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.u5 = UpSample(
            x_dim=2 * k, filters=k, concat=True, old_x_dim=k, act_param=act_param
        )

        H, W, k = 2 * H, 2 * W, k // 2
        self.u6 = DenseBlock(
            x_shape=(2 * k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.u7 = UpSample(
            x_dim=2 * k,
            filters=k,
            concat=True,
            old_x_dim=k,
            act_param=act_param,
            upsample=False,
        )

        self.u8 = DenseBlock(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=k,
            layers=denselayers,
        )

        self.u9 = UpSample(
            x_dim=k, filters=out_channels, upsample=False, activation=False
        )

        if g_out == "x":
            self.res_wt = 0.0
            self.ofunc = nn.Identity()
        elif g_out == "dx":
            self.res_wt = 1.0
            self.ofunc = nn.Identity()
        elif g_out == "bdx":
            self.res_wt = 1.0
            self.ofunc = nn.Sigmoid()
        # ------------------------------------------------

    def forward(self, input_x, input_z=None):
        x1 = self.d1(input_x=input_x)
        x1 = self.d2(input_x=x1)

        x2 = self.d3(input_x=x1)
        x2 = self.d4(input_x=x2, input_z=input_z)

        x3 = self.d5(input_x=x2)
        x3 = self.d6(input_x=x3, input_z=input_z)

        x4 = self.d7(input_x=x3)
        x4 = self.d8(input_x=x4, input_z=input_z)

        x5 = self.base(input_x=x4, input_z=input_z)

        x6 = self.u1(input_x=x5)
        x6 = self.u2(input_x=x6, input_z=input_z)

        x7 = self.u3(input_x=x6, old_x=x3)
        x7 = self.u4(input_x=x7, input_z=input_z)

        x8 = self.u5(input_x=x7, old_x=x2)
        x8 = self.u6(input_x=x8, input_z=input_z)

        x9 = self.u7(input_x=x8, old_x=x1)
        x9 = self.u8(input_x=x9, input_z=input_z)

        x10 = self.u9(input_x=x9)

        output = x10

        return output


class critic_dense_2x64x76_to_1x64x64(nn.Module):
    """Critic model using Denseblocks
    input_x and input_y are both assumed to have
    the shape (N, C, H, W)
    """

    def __init__(
        self, x_shape, channels_y, k0=24, act_param=0.1, denselayers=3, dense_int_out=16
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(critic_dense_2x64x76_to_1x64x64, self).__init__()
        C0_x, H0, W0 = x_shape
        C0_y = channels_y

        # ------ Convolution layers -----------------------------
        H, W = H0, W0
        self.cnn1 = DownSample(
            x_dim=C0_x + C0_y, filters=k0, downsample=False, act_param=act_param
        )

        self.cnn2 = DenseBlock(
            x_shape=(k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=k0,
            layers=denselayers,
        )

        self.cnn3 = DownSample(
            x_dim=k0, filters=2 * k0, act_param=act_param, ds_k=4, ds_s=4
        )
        H, W = (H - 2) // 4 + 1, (W - 2) // 4 + 1
        self.cnn4 = DenseBlock(
            x_shape=(2 * k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=2 * k0,
            layers=denselayers,
        )

        self.cnn5 = DownSample(
            x_dim=2 * k0, filters=4 * k0, act_param=act_param, ds_k=4, ds_s=4
        )
        H, W = (H - 2) // 4 + 1, (W - 2) // 4 + 1

        self.cnn6 = DenseBlock(
            x_shape=(4 * k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=4 * k0,
            layers=denselayers,
        )

        self.cnn7 = DownSample(x_dim=4 * k0, filters=8 * k0, act_param=act_param)
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1
        self.cnn8 = DenseBlock(
            x_shape=(8 * k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=8 * k0,
            layers=denselayers,
        )

        # ----- Dense layers------------------------------
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(in_features=8 * k0 * H * W, out_features=128)
        self.LReLU = nn.ELU(alpha=act_param)
        self.LN = ApplyNormalization(x_shape=(128), normalization="ln")
        self.lin2 = nn.Linear(in_features=128, out_features=128)
        self.lin3 = nn.Linear(in_features=128, out_features=1)

        # ------------------------------------------------

    def forward(self, input_x, input_y):
        xy = cat((input_x, input_y), dim=1)

        x = self.cnn1(input_x=xy)
        x = self.cnn2(input_x=x)
        x = self.cnn3(input_x=x)
        x = self.cnn4(input_x=x)
        x = self.cnn5(input_x=x)
        x = self.cnn6(input_x=x)
        x = self.cnn7(input_x=x)
        x = self.cnn8(input_x=x)

        x = self.flat(x)
        x = self.lin1(x)
        x = self.LReLU(x)
        x = self.LN(x)
        x = self.lin2(x)
        x = self.LReLU(x)
        x = self.LN(x)
        output = self.lin3(x)

        return output
