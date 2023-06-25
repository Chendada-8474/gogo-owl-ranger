import argparse


def restri_batch_size(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a integer" % (x,))

    if x < 1:
        raise argparse.ArgumentTypeError("%r have to be bigger than 0" % (x,))
    return x


def restri_threshold(arg):
    try:
        arg = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if arg <= 0 or arg >= 1:
        raise argparse.ArgumentTypeError("Argment must be between 0 and 1")
    return arg


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="path of model", required=True)
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="path of wav file or a folder",
        required=True,
    )
    parser.add_argument(
        "-b", "--batch", default=1, type=restri_batch_size, help="batch size"
    )
    parser.add_argument(
        "-md",
        "--min-duration",
        type=float,
        help="minimum positive detection duration.",
        default=0.05,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.5,
        type=restri_threshold,
        help="the threshold of confidence of target detected, dedault as 0.5",
    )

    args = parser.parse_args()
    return args
