import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data_x, data_y, args):
        self.data_x = data_x
        self.data_y = data_y
        self.args = args

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        x, y = self.data_x, self.data_y[idx]
        return x, y + torch.randn_like(y) * self.args.noiselvl


def get_heat_eq_single_source_dataset(args):
    y = (
        torch.tensor(
            np.load(
                "/scratch1/murgoiti/HeatEquationSingleSource_GenerateSamples/y_list_1.npy"
            )
        )
        .float()
        .squeeze()
        .unsqueeze(1)
    )
    if args.xtype == "vector":
        x = (
            torch.tensor(
                np.load(
                    "/scratch1/murgoiti/HeatEquationSingleSource_GenerateSamples/x_vect_list_1.npy"
                )
            )
            .float()
            .squeeze()
        )
    elif args.xtype == "image":
        x = (
            torch.tensor(
                np.load(
                    "/scratch1/murgoiti/HeatEquationSingleSource_GenerateSamples/x_list_1.npy"
                )
            )
            .float()
            .squeeze()
            .unsqueeze(1)
        )

    n = len(x)
    x_train, y_train = x[: int(n * 0.8)], y[: int(n * 0.8)]
    x_val, y_val = x[int(n * 0.8) : int(n * 0.9)], y[int(n * 0.8) : int(n * 0.9)]
    x_test, y_test = x[int(n * 0.9) : int(n * 1.0)], y[int(n * 0.9) : int(n * 1.0)]
    return (
        CustomDataset(x_train, y_train, args),
        CustomDataset(x_val, y_val, args),
        CustomDataset(x_test, y_test, args),
    )


def get_heat_eq_double_source_dataset(args):

    y = []
    x = []
    for i in range(1, 6):
        y.append(
            np.load(
                f"/scratch1/murgoiti/HeatEquationDoubleSource_GenerateSamples/y_list_{i}.npy"
            )
        )
        if args.xtype == "vector":
            x.append(
                np.load(
                    f"/scratch1/murgoiti/HeatEquationDoubleSource_GenerateSamples/x_vect_list_{i}.npy"
                )
            )
        elif args.xtype == "image":
            x.append(
                np.load(
                    f"/scratch1/murgoiti/HeatEquationDoubleSource_GenerateSamples/x_list_{i}.npy"
                )
            )

    y = torch.tensor(np.concatenate(y, axis=0)).float().squeeze().unsqueeze(1)
    x = torch.tensor(np.concatenate(x, axis=0)).float().squeeze()
    if args.xtype == "image":
        x = x.unsqueeze(1)

    n = len(x)
    x_train, y_train = x[: int(n * 0.8)], y[: int(n * 0.8)]
    x_val, y_val = x[int(n * 0.8) : int(n * 0.9)], y[int(n * 0.8) : int(n * 0.9)]
    x_test, y_test = x[int(n * 0.9) : int(n * 1.0)], y[int(n * 0.9) : int(n * 1.0)]
    return (
        CustomDataset(x_train, y_train, args),
        CustomDataset(x_val, y_val, args),
        CustomDataset(x_test, y_test, args),
    )


def get_helmholtz_eq_dataset(args):

    y = []
    x = []
    for i in range(1, 11):
        y.append(
            np.load(f"/scratch1/murgoiti/HelmholtzEq_GenerateSamples/y_list_{i}.npy")
        )
        if args.xtype == "vector":
            x.append(
                np.load(
                    f"/scratch1/murgoiti/HelmholtzEq_GenerateSamples/mu_vect_list_{i}.npy"
                )
            )
        elif args.xtype == "image":
            x.append(
                np.load(
                    f"/scratch1/murgoiti/HelmholtzEq_GenerateSamples/mu_list_{i}.npy"
                )
            )

    y = torch.tensor(np.concatenate(y, axis=0)).float().squeeze().unsqueeze(1)
    x = torch.tensor(np.concatenate(x, axis=0)).float().squeeze()
    if args.xtype == "image":
        x = x.unsqueeze(1)

    n = len(x)
    x_train, y_train = x[: int(n * 0.8)], y[: int(n * 0.8)]
    x_val, y_val = x[int(n * 0.8) : int(n * 0.9)], y[int(n * 0.8) : int(n * 0.9)]
    x_test, y_test = x[int(n * 0.9) : int(n * 1.0)], y[int(n * 0.9) : int(n * 1.0)]
    return (
        CustomDataset(x_train, y_train, args),
        CustomDataset(x_val, y_val, args),
        CustomDataset(x_test, y_test, args),
    )
