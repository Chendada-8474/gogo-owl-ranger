import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import GoGoDataset
from utils.model import CRNN
from utils.config import *
from utils.tools import slice_piece, PrograssBar
from utils.evaluate import evaluate


TARGET_SR = pre_prosessing_config["target_sample_rate"]
SAMPLE_DURATION = pre_prosessing_config["sample_duration"]
SAMPLE_HOP = pre_prosessing_config["sample_hop"]

EPOCHS = training_config["epochs"]
BATCH_SIZE = training_config["batch_size"]
LR = training_config["learning_rate"]
DATASET_NAME = training_config["dataset"]
SKIP_FALSE_SAMPLE = training_config["skip_false_sample"]

HOP = mel_specrogram_config["n_fft"] // 2

progbar = PrograssBar()


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
    print(progbar.title)
    for i in range(EPOCHS):
        sum_loss, num_loss = 0, 0

        accuracy, percision, recall, f1_score = evaluate(model, val_loader, piece_width)

        for j, (sample, label) in enumerate(train_loader):

            start_time_points = slice_piece(
                sample.shape[3], piece_width, sample_hop, shuffle=True
            )
            for s in start_time_points:
                piece_sample = sample[:, :, :, s : s + piece_width + 1]
                piece_label = label[:, s : s + piece_width + 1]
                if SKIP_FALSE_SAMPLE and all(l.item() == 0 for l in piece_label[0]):
                    continue
                optimiser.zero_grad()
                predictions = model(piece_sample)
                loss = loss_func(predictions, piece_label)

                sum_loss += loss.item()
                num_loss += 1
                loss.backward()
                optimiser.step()

                progbar.progress_bar(
                    j,
                    len(train_loader),
                    loss.item(),
                    i + 1,
                    EPOCHS,
                    accuracy=accuracy,
                    percision=percision,
                    recall=recall,
                    f1_score=f1_score,
                )

        progbar.line_break()

        epoch_loss = sum_loss / num_loss
        accuracy, percision, recall, f1_score = evaluate(model, val_loader, piece_width)

        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy
        training_indicator.append(
            [i + 1, epoch_loss, accuracy, percision, recall, f1_score]
        )

    training_indicator = pd.DataFrame(
        np.array(training_indicator),
        columns=["epoch", "loss", "accuracy", "percision", "recall", "f1_score"],
    )

    return model, best_model, training_indicator


def save_model(model, save_to=""):
    pass


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = GoGoDataset(
        DATASET_NAME,
        TARGET_SR,
        mode="train",
        device=device,
    )
    val_set = GoGoDataset(
        DATASET_NAME,
        TARGET_SR,
        mode="val",
        device=device,
    )

    import matplotlib.pyplot as plt
    import librosa

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

    # torch.save(last_model, "./models/last.pt")


if __name__ == "__main__":
    main()
