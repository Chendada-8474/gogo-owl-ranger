import torch
from torch.utils.data import DataLoader
from utils.tools import ConfusionMatrix, slice_piece
from utils.model import CRNN


def evaluate(model: CRNN, dataloader: DataLoader, piece_width):

    confusion_matrix = ConfusionMatrix()
    model.eval()

    for sample, labels in dataloader:
        start_time_points = slice_piece(
            sample.shape[3], piece_width, piece_width, shuffle=False, make_up=True
        )
        for s in start_time_points:
            piece = sample[:, :, :, s : s + piece_width + 1]
            label = labels[:, s : s + piece_width + 1]
            predictions = model(piece)
            for pred, ground_truth in zip(predictions.permute(0, 2, 1)[0], label[0]):
                pred = (pred == torch.max(pred)).nonzero(as_tuple=True)[0].item()
                confusion_matrix.judge(pred, ground_truth)
    confusion_matrix.summary()
    accuracy = confusion_matrix.accuracy
    percision = confusion_matrix.percision
    recall = confusion_matrix.recall
    f1_score = confusion_matrix.f1_score
    return accuracy, percision, recall, f1_score


if __name__ == "__main__":
    from dataset import GoGoDataset
    from config import *
    import torchaudio

    TARGET_SR = pre_prosessing_config["target_sample_rate"]
    SAMPLE_DURATION = pre_prosessing_config["sample_duration"]
    SAMPLE_HOP = pre_prosessing_config["sample_hop"]

    EPOCHS = training_config["epochs"]
    BATCH_SIZE = training_config["batch_size"]
    LR = training_config["learning_rate"]
    DATASET_NAME = training_config["dataset"]

    HOP = mel_specrogram_config["n_fft"] // 2
    piece_width = TARGET_SR * SAMPLE_DURATION // HOP + 1

    model = CRNN()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=mel_specrogram_config["n_fft"],
        n_mels=mel_specrogram_config["n_mels"],
        f_max=mel_specrogram_config["f_max"],
        f_min=mel_specrogram_config["f_min"],
    )
    val_set = GoGoDataset(
        "gogo-owl",
        16000,
        transform,
        mode="val",
        device="cpu",
    )
    print(val_set[0])
    print(len(val_set))
    print(val_set[0][0].size())
    print(piece_width)

    val_loader = DataLoader(dataset=val_set, batch_size=5, shuffle=True)
    print(evaluate(model, val_loader, piece_width))
