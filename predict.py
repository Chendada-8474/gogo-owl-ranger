import os
import torch
import yaml
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path, PurePath
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from utils.model import CRNN
from utils.dataset import PredictDataset
from utils.tools import slice_piece, get_device_info


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
        help="path of image, vedeo or a folder",
        required=True,
    )
    parser.add_argument(
        "-b", "--batch", type=restri_batch_size, help="batch size", required=True
    )
    parser.add_argument(
        "-i", "--interval", type=float, help="interval of probability", default=0.5
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.5,
        type=restri_threshold,
        help="the threshold of probability of target detected",
    )

    args = parser.parse_args()
    return args


def read_predict_config(model_path) -> dict:
    info_path = PurePath(Path(model_path).parent, Path("model_info.yaml"))
    with open(info_path, "r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)
    return model_info


def predict(model, loader, model_info, device="cpu"):
    target_sr = model_info["target_sample_rate"]
    sample_duration = model_info["sample_duration"]
    hop = model_info["n_fft"] // 2
    piece_width = target_sr * sample_duration // hop + 1

    proba_seq = defaultdict(list)

    for audio, paths in tqdm(loader):
        audio.to(device)
        start_points = slice_piece(
            audio.shape[3], piece_width, piece_width, shuffle=False, make_up=True
        )
        for i, s in enumerate(start_points):
            audio_piece = audio[:, :, :, s : s + piece_width]
            prediction = model(audio_piece)
            for pred, path in zip(prediction.squeeze(1), paths):
                filename = os.path.basename(path)
                if i < len(start_points) - 1:
                    proba_seq[filename] += torch.round(pred, decimals=4).tolist()
                else:
                    res_start = audio.shape[3] - len(proba_seq[filename])
                    res_pre = torch.round(pred, decimals=4).tolist()[-res_start:]
                    proba_seq[filename] += res_pre
    return proba_seq


def target_coverage(
    seq: list, threshold: float, sample_rate: int, mel_hop: int
) -> list:
    piece_sec = 1 / sample_rate * mel_hop
    true_count = sum([1 for p in seq if p > threshold])
    coverage = round(true_count * piece_sec, 2)
    proportion = round(true_count / len(seq), 3)
    return coverage, proportion


def extract_max_in_interval(predictions, interval, model_info) -> list:
    target_sr = model_info["target_sample_rate"]
    sample_duration = model_info["sample_duration"]
    hop = model_info["n_fft"] // 2
    piece_width = target_sr * sample_duration // hop + 1
    interval_len = piece_width / sample_duration * interval
    half_interval_len = int(interval_len // 2)
    time_points = [
        n * interval_len for n in range(int(len(predictions) // interval_len))
    ]
    results = []
    for tp in time_points:
        s = int(round(tp)) - half_interval_len
        e = int(round(tp)) + half_interval_len
        if s < 0:
            s = 0
        if e > len(predictions):
            e = len(predictions)
        results.append(round(max(predictions[s:e]), 4))
    return results


def save_result(
    result: pd.DataFrame,
    save_path,
    filename="results",
    time_stemp=True,
):
    ext = ".csv"
    if time_stemp:
        time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        filename = "%s_%s" % (filename, time_now)
    filename += ext
    if not os.path.isdir(save_path):
        save_path = Path(save_path).parent
    save_path = PurePath.joinpath(Path(save_path), Path(filename))
    result.to_csv(save_path, index=False)
    print("results have been save to %s" % save_path)


def main():
    opt = parse_opt()
    model_path = opt.model
    source_path = opt.source
    batch_size = opt.batch
    interval = opt.interval
    threshold = opt.threshold

    model_info = read_predict_config(model_path)

    target_sr = model_info["target_sample_rate"]
    mel_hop = model_info["n_fft"] // 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_info = get_device_info(device)
    print(device_info)

    model = CRNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predict_set = PredictDataset(source_path, model_info, device=device)

    predict_loader = DataLoader(
        dataset=predict_set, batch_size=batch_size, shuffle=False
    )
    predictions = predict(model, predict_loader, model_info, device=device)

    proba_seq = defaultdict(list)
    targ_cov = defaultdict(list)

    for k, v in predictions.items():
        extract = extract_max_in_interval(v, interval, model_info)
        proba_seq["file_name"] += [k] * len(extract)
        proba_seq["time_s"] += [round(interval * i, 1) for i in range(len(extract))]
        proba_seq["probability"] += extract

        cov, prop = target_coverage(v, threshold, target_sr, mel_hop)
        targ_cov["file_name"].append(k)
        targ_cov["coverage"].append(cov)
        targ_cov["proportion"].append(prop)

    save_result(pd.DataFrame(proba_seq), source_path, filename="probabiliy_sequence")
    save_result(pd.DataFrame(targ_cov), source_path, filename="target_coverage")


if __name__ == "__main__":
    main()
