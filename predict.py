import os
import torch
import yaml
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path, PurePath
from tqdm import tqdm
from utils.model import CRNN
from utils.dataset import PredictDataset
from collections import defaultdict
from utils.tools import slice_piece
from datetime import datetime


def restri_batch_size(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a integer" % (x,))

    if x < 1:
        raise argparse.ArgumentTypeError("%r have to be bigger than 0" % (x,))
    return x


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

    results = defaultdict(list)

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
                    results[filename] += torch.round(pred, decimals=4).tolist()
                else:
                    res_start = audio.shape[3] - len(results[filename])
                    res_pre = torch.round(pred, decimals=4).tolist()[-res_start:]
                    results[filename] += res_pre
    return results


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


def save_result(result: pd.DataFrame, save_path, time_stemp=True):
    filename = "result"
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
    model_info = read_predict_config(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CRNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predict_set = PredictDataset(source_path, model_info, device=device)

    predict_loader = DataLoader(
        dataset=predict_set, batch_size=batch_size, shuffle=False
    )
    predictions = predict(model, predict_loader, model_info, device=device)

    results = {
        "file_name": [],
        "time_s": [],
        "probability": [],
    }

    for k, v in predictions.items():
        predictions[k] = extract_max_in_interval(v, interval, model_info)
        results["file_name"] += [k] * len(predictions[k])
        results["time_s"] += [interval * i for i in range(len(predictions[k]))]
        results["probability"] += predictions[k]

    save_result(pd.DataFrame(results), source_path)


if __name__ == "__main__":
    main()
