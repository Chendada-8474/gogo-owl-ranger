import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import GoGoDataset
from utils.model import CRNN
from utils.config import *
from utils.tools import slice_piece
from utils.evaluate import evaluate


TARGET_SR = pre_prosessing_config["target_sample_rate"]
SAMPLE_DURATION = pre_prosessing_config["sample_duration"]
SAMPLE_HOP = pre_prosessing_config["sample_hop"]

EPOCHS = training_config["epochs"]
BATCH_SIZE = training_config["batch_size"]
LR = training_config["learning_rate"]
DATASET_NAME = training_config["dataset"]

HOP = mel_specrogram_config["n_fft"] // 2


def train(
    model: CRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_func: torch.nn.CrossEntropyLoss,
    optimiser: torch.optim.Adam,
    device="cpu",
):
    piece_width = TARGET_SR * SAMPLE_DURATION // HOP + 1
    sample_hop = SAMPLE_HOP * HOP
    best_model, best_accuracy = model, 0
    training_indicator = []
    model.train()
    for i in range(EPOCHS):
        print(f"epoch {i + 1}/{EPOCHS}")
        for sample, label in tqdm(train_loader):
            start_time_points = slice_piece(
                sample.shape[3], piece_width, sample_hop, shuffle=True
            )
            for s in start_time_points:
                piece_sample = sample[:, :, :, s : s + piece_width + 1]
                piece_label = label[:, s : s + piece_width + 1]
                if all(l.item() == 0 for l in piece_label[0]):
                    continue
                optimiser.zero_grad()
                predictions = model(piece_sample)
                loss = loss_func(predictions, piece_label)
                loss.backward()
                optimiser.step()
        accuracy, percision, recall, f1_score = evaluate(model, val_loader, piece_width)

        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy
        training_indicator.append([i + 1, accuracy, percision, recall, f1_score])

    training_indicator = pd.DataFrame(
        np.array(training_indicator),
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

    train_set = GoGoDataset(
        DATASET_NAME,
        TARGET_SR,
        transform,
        mode="train",
        device=device,
    )
    val_set = GoGoDataset(
        DATASET_NAME,
        TARGET_SR,
        transform,
        mode="val",
        device=device,
    )

    # import matplotlib.pyplot as plt
    # import librosa

    # fig, axs = plt.subplots(2)
    # axs[0].plot(train_set[2][1])
    # axs[1].imshow(
    #     librosa.power_to_db(train_set[2][0][0]), origin="lower", aspect="auto"
    # )
    # plt.show()

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CRNN()
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    last_model, best_model, training_indicator = train(
        model, train_loader, val_loader, loss_func, optimiser, device=device
    )

    print(training_indicator)


if __name__ == "__main__":
    main()
