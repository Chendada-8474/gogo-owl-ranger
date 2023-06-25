import os
import sys
import torch
import random


class ConfusionMatrix:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.matrix = [[0, 0], [0, 0]]
        self.accuracy = None
        self.percision = None
        self.recall = None
        self.f1_score = None

    def __str__(self):
        return (
            "confision matrix%s\naccuracy: %s\npercision: %s\nrecall: %s\nf1_score: %s"
            % (self.matrix, self.accuracy, self.percision, self.recall, self.f1_score)
        )

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def judge(self, prediction: int, ground_truth: int):
        if prediction and ground_truth:
            self.tp += 1
        elif prediction and not ground_truth:
            self.fp += 1
        elif not prediction and ground_truth:
            self.fn += 1
        elif not prediction and not ground_truth:
            self.tn += 1

        self._update_matrix()

    def _update_matrix(self):
        self.matrix = [[self.tp, self.fp], [self.fn, self.tn]]

    def summary(self):
        num_sample = sum((self.tp, self.fp, self.fn, self.tn))
        num_positive = self.tp + self.fp
        num_true_positive = self.tp + self.fn
        self.accuracy = (self.tp + self.tn) / num_sample

        if num_positive:
            self.percision = self.tp / num_positive
        if num_true_positive:
            self.recall = self.tp / num_true_positive

        if self.percision and self.recall:
            pr = self.percision + self.recall
            self.f1_score = (2 * self.percision * self.recall) / pr

    @property
    def results(self):
        return self.accuracy, self.percision, self.recall, self.f1_score


def skip_false_sample(label, skip_rate: float = 0.0):
    return all(l.item() == 0 for l in label[0]) and random.random() < skip_rate


class PrograssBar:
    @staticmethod
    def training(i, num_sample, loss, epoch, num_epoch, bar_length=30):
        i += 1
        loss = round(loss, 4) if loss else str(loss)
        progress = "█" * ((i * bar_length) // num_sample)
        unprogess = " " * (bar_length - (i * bar_length) // num_sample)
        percent = "%s/%s" % (i, num_sample)
        loss = PrograssBar._fill_zero(loss)
        print(
            f"\rloss: {loss} [{progress}{unprogess}] {percent}   epoch {epoch}/{num_epoch}",
            end="",
        )

    @staticmethod
    def evaluate(accuracy, percision, recall, f1_score):
        if accuracy:
            accuracy = round(accuracy, 4)
        if percision:
            percision = round(percision, 4)
        if recall:
            recall = round(recall, 4)
        if f1_score:
            f1_score = round(f1_score, 4)
        accuracy = PrograssBar._fill_zero(accuracy)
        percision = PrograssBar._fill_zero(percision)
        recall = PrograssBar._fill_zero(recall)
        f1_score = PrograssBar._fill_zero(f1_score)
        print(
            f"\naccuracy: {accuracy} percision: {percision} recall: {recall} f1_score: {f1_score}"
        )

    @staticmethod
    def line_break():
        print("\r")

    def _fill_zero(string, l=8):
        return "%s%s" % (str(string), " " * (l - len(str(string))))


def get_device_info(device):
    torch_version = "%s %s" % ("torch", torch.__version__)
    device_name = torch.cuda.get_device_name() if device == "cuda" else None
    device_number = torch.cuda.current_device() if device == "cuda" else None
    return f"{torch_version} CUDA:{device_number} {device_name}"


if __name__ == "__main__":
    pass
