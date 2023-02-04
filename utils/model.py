import torch.nn as nn
from utils.config import pre_prosessing_config
from torchvision.models.detection.backbone_utils import mobilenet_backbone


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        self.cnn = self._cnn_backbone()
        self.map_to_seq = nn.Linear(96, 96)
        self.rnn1 = nn.LSTM(96, 96, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(192, 96, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(192, 1)
        self.softmax = nn.Sigmoid()

    def _cnn_backbone(self):

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

        cnn_backbone.add_module("ReLu0", leaky_relu)
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

        cnn_backbone.add_module("ReLu1", leaky_relu)
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

        cnn_backbone.add_module("batchnorm0", nn.BatchNorm2d(num_channel))
        cnn_backbone.add_module("ReLu2", leaky_relu)
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

        cnn_backbone.add_module("batchnorm1", nn.BatchNorm2d(num_channel))
        cnn_backbone.add_module("ReLu3", leaky_relu)
        cnn_backbone.add_module("MaxPool2d3", nn.MaxPool2d(kernel_size=(2, 1)))

        return cnn_backbone

    def forward(self, sample):
        sample = self.cnn(sample)
        batch_size, channel, height, width = sample.size()
        sample = sample.view(batch_size, channel * height, width)
        sample = sample.permute(0, 2, 1)
        sample = self.map_to_seq(sample)
        sample, _ = self.rnn1(sample)
        sample, _ = self.rnn2(sample)
        logits = self.dense(sample)
        predictions = self.softmax(logits)
        predictions = predictions.permute(0, 2, 1)
        return predictions


if __name__ == "__main__":
    import torchaudio
    import torch
    from dataset import GoGoDataset
    from config import pre_prosessing_config, mel_specrogram_config
    from tools import ConfusionMatrix
    from torch.utils.data import DataLoader

    TARGET_SAMPLE_RATE = pre_prosessing_config["target_sample_rate"]

    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=mel_specrogram_config["n_fft"],
        n_mels=mel_specrogram_config["n_mels"],
        f_max=mel_specrogram_config["f_max"],
        f_min=mel_specrogram_config["f_min"],
    )

    device = "cude" if torch.cuda.is_available() else "cpu"

    gogo = GoGoDataset(
        "gogo-owl", TARGET_SAMPLE_RATE, transformation, device=device, mode="val"
    )
    loader = DataLoader(dataset=gogo, batch_size=1, shuffle=False)
    model = CRNN()
    for sample, label in loader:
        sample = sample[:, :, :, :313]
        output = model.forward(sample)
        # cm = ConfusionMatrix()
        # for o, l in zip(output, gogo[0][1][:313]):
        #     pred = (o == torch.max(o)).nonzero(as_tuple=True)[0].item()
        #     print(pred, l, o)
        #     cm.judge(pred, l)
        # cm.summary()
        # print(cm)
