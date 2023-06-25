import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        self.backbone = self._cnn_backbone()
        self.rnn1 = nn.GRU(640, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def _cnn_backbone(self):
        dropout = nn.Dropout(0.5)
        leaky_relu = nn.ReLU()
        cnn_backbone = nn.Sequential()
        cnn_backbone.add_module(
            "cnn0",
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("MaxPool2d0", nn.MaxPool2d(kernel_size=(2, 1)))
        cnn_backbone.add_module("batchnorm0", nn.BatchNorm2d(128))
        cnn_backbone.add_module("ReLu0", leaky_relu)
        cnn_backbone.add_module("dropout0", dropout)

        cnn_backbone.add_module(
            "cnn1",
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("MaxPool2d1", nn.MaxPool2d(kernel_size=(2, 1)))
        cnn_backbone.add_module("batchnorm1", nn.BatchNorm2d(128))
        cnn_backbone.add_module("ReLu1", leaky_relu)
        cnn_backbone.add_module("dropout1", dropout)

        cnn_backbone.add_module(
            "cnn2",
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("MaxPool2d1", nn.MaxPool2d(kernel_size=(2, 1)))
        cnn_backbone.add_module("batchnorm1", nn.BatchNorm2d(128))
        cnn_backbone.add_module("ReLu1", leaky_relu)
        cnn_backbone.add_module("dropout1", dropout)

        cnn_backbone.add_module(
            "cnn2",
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        cnn_backbone.add_module("MaxPool2d2", nn.MaxPool2d(kernel_size=(2, 1)))
        cnn_backbone.add_module("batchnorm2", nn.BatchNorm2d(128))
        cnn_backbone.add_module("ReLu2", leaky_relu)
        cnn_backbone.add_module("dropout2", dropout)

        cnn_backbone.add_module(
            "cnn3",
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        cnn_backbone.add_module("MaxPool2d3", nn.MaxPool2d(kernel_size=(2, 1)))
        cnn_backbone.add_module("batchnorm3", nn.BatchNorm2d(128))
        cnn_backbone.add_module("ReLu3", leaky_relu)
        cnn_backbone.add_module("dropout3", dropout)

        return cnn_backbone

    def forward(self, sample):
        sample = self.backbone(sample)
        sample = sample.flatten(1, 2)
        sample = sample.permute(0, 2, 1)
        sample, _ = self.rnn1(sample)
        sample, _ = self.rnn2(sample)
        logits = self.dense(sample)
        predictions = self.sigmoid(logits)
        predictions = predictions.permute(0, 2, 1)
        return predictions


if __name__ == "__main__":
    pass
