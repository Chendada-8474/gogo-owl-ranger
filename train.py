import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils.dataset import GoGoDataset
from utils.model import CRNN
from utils.config import training_config, mel_specrogram_config
from utils.tools import slice_piece
from utils.evaluate import evaluate


EPOCHS = training_config["epochs"]
BATCH_SIZE = training_config["batch_size"]
LR = training_config["learning_rate"]
DATASET_NAME = training_config["dataset"]
VAL_PROPORTION = training_config["val_proportion"]

TARGET_SR = mel_specrogram_config["target_sample_rate"]
SAMPLE_DURATION = mel_specrogram_config["sample_duration"]
HOP = mel_specrogram_config["n_fft"] // 2
SAMPLE_HOP = mel_specrogram_config["sample_hop"]


def train(
    model: CRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_func: torch.nn.CrossEntropyLoss,
    optimiser: torch.optim.Adam,
    device="cpu",
):
    piece_width = TARGET_SR * SAMPLE_DURATION // HOP + 1
    sample_hop = SAMPLE_HOP // HOP

    best_model, best_accuracy = model, 0
    training_indicator = np.array([])

    for i in range(EPOCHS):
        print(f"epoch {i}/{EPOCHS}")

        for sample, label in tqdm(train_loader):
            start_time_points = slice_piece(
                sample.shape[2], piece_width, sample_hop, shuffle=True
            )

            for s in start_time_points:
                piece_sample = sample[:, :, s : s + piece_width + 1]
                piece_label = label[s : s + piece_width + 1]

                predictions = model(piece_sample)
                loss = loss_func(predictions, piece_label)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        accuracy, percision, recall, f1_score = evaluate(model, val_loader, piece_width)

        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy

        training_indicator.append([i, accuracy, percision, recall, f1_score])

    training_indicator = pd.DataFrame(
        training_indicator,
        columns=["epoch", "accuracy", "percision", "recall", "f1_score"],
    )

    return model, best_model, training_indicator


def main():

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=mel_specrogram_config["n_fft"],
        n_mels=mel_specrogram_config["n_mels"],
        f_max=mel_specrogram_config["f_max"],
        f_min=mel_specrogram_config["f_min"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gogo_dataset = GoGoDataset(
        DATASET_NAME,
        TARGET_SR,
        transform,
        device=device,
    )

    num_val = len(gogo_dataset) * VAL_PROPORTION
    num_val = int(num_val) if num_val >= 1 else 1
    train_set, val_set = random_split(
        gogo_dataset, (len(gogo_dataset) - num_val, num_val)
    )

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    model = CRNN()
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    last_model, best_model, training_indicator = train(
        model, train_loader, val_loader, loss_func, optimiser, device=device
    )


if __name__ == "__main__":
    main()
