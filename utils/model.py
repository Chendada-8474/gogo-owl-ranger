import torch.nn as nn
from utils.config import pre_prosessing_config
from torchvision.models.detection.backbone_utils import mobilenet_backbone


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        self.backbone = self._cnn_backbone()
        self.map_to_seq = nn.Linear(96, 96)
        self.rnn1 = nn.LSTM(96, 96, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(192, 96, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(192, 1)
        self.sigmoid = nn.Sigmoid()

    def _cnn_backbone(self):

        dropout = nn.Dropout(0.25)
        leaky_relu = nn.LeakyReLU()
        cnn_backbone = nn.Sequential()
        num_channel = 96
        cnn_backbone.add_module(
            "cnn0",
            nn.Conv2d(
                in_channels=1,
                out_channels=num_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("batchnorm0", nn.BatchNorm2d(num_channel))
        cnn_backbone.add_module("ReLu0", leaky_relu)
        cnn_backbone.add_module("dropout0", dropout)
        cnn_backbone.add_module("MaxPool2d0", nn.MaxPool2d(kernel_size=(4, 1)))

        cnn_backbone.add_module(
            "cnn1",
            nn.Conv2d(
                in_channels=num_channel,
                out_channels=num_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("batchnorm1", nn.BatchNorm2d(num_channel))
        cnn_backbone.add_module("ReLu1", leaky_relu)
        cnn_backbone.add_module("dropout1", dropout)
        cnn_backbone.add_module("MaxPool2d1", nn.MaxPool2d(kernel_size=(4, 1)))

        cnn_backbone.add_module(
            "cnn2",
            nn.Conv2d(
                in_channels=num_channel,
                out_channels=num_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        cnn_backbone.add_module("batchnorm2", nn.BatchNorm2d(num_channel))
        cnn_backbone.add_module("ReLu2", leaky_relu)
        cnn_backbone.add_module("dropout2", dropout)
        cnn_backbone.add_module("MaxPool2d2", nn.MaxPool2d(kernel_size=(2, 1)))

        cnn_backbone.add_module(
            "cnn3",
            nn.Conv2d(
                in_channels=num_channel,
                out_channels=num_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        cnn_backbone.add_module("batchnorm3", nn.BatchNorm2d(num_channel))
        cnn_backbone.add_module("ReLu3", leaky_relu)
        cnn_backbone.add_module("dropout3", dropout)
        cnn_backbone.add_module("MaxPool2d3", nn.MaxPool2d(kernel_size=(2, 1)))

        return cnn_backbone

    def forward(self, sample):
        sample = self.backbone(sample)
        batch_size, channel, height, width = sample.size()
        sample = sample.view(batch_size, channel * height, width)
        sample = sample.permute(0, 2, 1)
        sample = self.map_to_seq(sample)
        sample, _ = self.rnn1(sample)
        sample, _ = self.rnn2(sample)
        logits = self.dense(sample)
        predictions = self.sigmoid(logits)
        predictions = predictions.permute(0, 2, 1)
        return predictions


if __name__ == "__main__":
    pass
