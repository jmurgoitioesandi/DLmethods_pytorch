import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(
        description="list of arguments", formatter_class=formatter
    )

    # Data parameters
    parser.add_argument(
        "--train_file",
        type=str,
        default="ONH_samples_2500_20000/training_data.h5",
        help=textwrap.dedent("""Data file containing training data pairs"""),
    )
    parser.add_argument(
        "--saving_dir",
        type=str,
        default="cWGAN_092323",
        help=textwrap.dedent("""Directory to save files"""),
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=12000,
        help=textwrap.dedent(
            """Number of training samples to use. Cannot be more than that available."""
        ),
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=1e-4,
        help=textwrap.dedent("""Learning rate."""),
    )
    parser.add_argument(
        "--seed_no",
        type=int,
        default=1008,
        help=textwrap.dedent("""Set the random seed"""),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help=textwrap.dedent("""Batch size"""),
    )
    return parser.parse_args()
