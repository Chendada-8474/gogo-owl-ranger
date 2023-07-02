import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.model import CRNN
from src.data import GoSample, GoAnnotation
from src.tool import PrograssBar, ConfusionMatrix, skip_false_sample
from src.read import GoConfig

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from config import *

MODELS_DIR = os.path.join(ROOT_DIR, "models")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GoConfig()
# writer = SummaryWriter()


class Evaluator:
    confusion_matrix = ConfusionMatrix()
    columns = ["epoch", "accuracy", "percision", "recall", "f1_score"]
    records = []

    def __init__(self, val_loader) -> None:
        self.val_loader = val_loader

    def evaluate(self, model: CRNN):
        self.confusion_matrix.reset()
        model.eval()
        for sample, label in self.val_loader:
            piece_start = 0
            while piece_start + config.piece_width <= sample.shape[3]:
                sample_piece = sample[
                    :, :, :, piece_start : piece_start + config.piece_width
                ]
                label_piece = label[:, piece_start : piece_start + config.piece_width]
                preds = model(sample_piece)
                for pred, gt in zip(preds.permute(0, 2, 1)[0], label_piece[0]):
                    self.confusion_matrix.judge(round(pred.item()), gt)
                piece_start += config.piece_width

        self.confusion_matrix.summary()
        results = self.confusion_matrix.results
        self.records.append(list(results))
        return results

    def to_csv(self, path):
        records = []
        for i, record in enumerate(self.records):
            records.append([i + 1] + record)
        table = pd.DataFrame(np.array(records), columns=self.columns)
        table.to_csv(path, index=False)


class ModelSaver:
    @staticmethod
    def mkdir_indexing() -> str:
        new_model_dir = os.path.join(MODELS_DIR, config.model_name)
        index = 1

        while os.path.isdir(new_model_dir):
            new_model_dir = "%s%s" % (new_model_dir, index)
            index += 1

        os.mkdir(new_model_dir)
        return new_model_dir

    @staticmethod
    def save_model(model: CRNN, path: str):
        torch.save(model.state_dict(), path)

    @staticmethod
    def save_model_info(path):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w") as info:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_name = torch.cuda.current_device() if device == "cuda" else "cpu"
            info.write("train_datetime: %s\n" % current_datetime)
            info.write("torch: %s\n" % torch.__version__)
            info.write("device: %s\n" % device_name)
            for k, v in pre_prosessing_config.items():
                info.write("%s: %s\n" % (k, v))
            for k, v in mel_specrogram_config.items():
                info.write("%s: %s\n" % (k, v))
            for k, v in training_config.items():
                info.write("%s: %s\n" % (k, v))


class GoTrain:
    EPOCHS = training_config["epochs"]
    epoch_now = None
    best_model = None
    best_f1 = 0

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: CRNN,
        loss_func: torch.nn.BCELoss,
        optimiser: torch.optim.Adam,
    ) -> None:
        self.model = model
        self.loss_func = loss_func
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = Evaluator(self.train_loader)

    def tune(self):
        for i in range(self.EPOCHS):
            self.model.train()
            self.epoch_now = i
            for j, (sample, annotation) in enumerate(self.train_loader):
                sample.to(device)
                annotation.to(device)
                loss = self._tune_file(sample, annotation)
                if not loss:
                    continue
                PrograssBar.training(
                    j, len(self.train_loader), loss, i + 1, self.EPOCHS
                )

            accuracy, percision, recall, f1_score = self.evaluator.evaluate(self.model)
            # writer.add_scalar("accuracy", accuracy, i)
            # writer.add_scalar("loss", loss, i)
            # writer.add_scalar("f1_score", f1_score, i)
            PrograssBar.evaluate(accuracy, percision, recall, f1_score)

            if f1_score and f1_score > self.best_f1:
                self.best_f1 = f1_score
                self.best_model = self.model

    def _tune_file(self, sample: GoSample, label: GoAnnotation):
        piece_width = config.piece_width
        piece_hop = config.piece_hop

        piece_start = 0
        losses = []

        while piece_start + piece_width <= sample.shape[3]:
            sample_piece = sample[:, :, :, piece_start : piece_start + piece_width]
            label_piece = label[:, piece_start : piece_start + piece_width]
            loss = self._tune_piece(sample_piece, label_piece)
            piece_start += piece_hop
            if loss:
                losses.append(loss.item())

        return sum(losses) / len(losses) if losses else None

    def _tune_piece(self, sample_piece, label_piece):
        if skip_false_sample(label_piece, skip_rate=config.skip_false_rate):
            return
        self.optimiser.zero_grad()
        pred = self.model(sample_piece)
        loss = self.loss_func(
            pred.to(device), label_piece.to(device).float().unsqueeze(1)
        )
        loss.backward()
        self.optimiser.step()
        return loss

    def save(self):
        dirname = ModelSaver.mkdir_indexing()
        log_path = os.path.join(dirname, "train_log.csv")
        model_path = os.path.join(dirname, "best.pth")
        model_info_path = os.path.join(dirname, "model_info.yaml")
        self.evaluator.to_csv(log_path)
        ModelSaver.save_model(self.model, model_path)
        ModelSaver.save_model_info(model_info_path)
