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
            piece = sample[:, :, :, s : s + piece_width]
            label = labels[:, s : s + piece_width]
            predictions = model(piece)
            for pred, ground_truth in zip(predictions.permute(0, 2, 1)[0], label[0]):
                pred = round(pred.item())
                confusion_matrix.judge(pred, ground_truth)

    confusion_matrix.summary()
    accuracy = confusion_matrix.accuracy
    percision = confusion_matrix.percision
    recall = confusion_matrix.recall
    f1_score = confusion_matrix.f1_score
    return accuracy, percision, recall, f1_score


if __name__ == "__main__":
    pass
