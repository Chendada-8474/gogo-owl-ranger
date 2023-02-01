import random


class ConfusionMatrix:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.matrix = [[self.tp, self.fp], [self.fn, self.tn]]
        self.accuracy = None
        self.percision = None
        self.recall = None
        self.f1_score = None

    def __str__(self):
        return (
            "confision matrix%s\naccuracy: %s\npercision: %s\nrecall: %s\nf1_score: %s"
            % (self.matrix, self.accuracy, self.percision, self.recall, self.f1_score)
        )

    def judge(self, prediction: int, ground_truth: int):

        assert (prediction == 0 or prediction == 1) and (
            ground_truth == 0 or ground_truth == 1
        )
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


def slice_piece(
    sample_width, piece_width, sample_hop, shuffle: bool = True, make_up=False
) -> list:
    num_piece = (sample_width - piece_width) // sample_hop
    start_time_points = [n * sample_hop for n in range(num_piece)]
    if shuffle:
        random.shuffle(start_time_points)
    res = (sample_width - piece_width) % sample_hop
    if make_up and res:
        start_time_points.append(sample_width - sample_hop)
    return start_time_points
