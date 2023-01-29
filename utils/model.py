import torch.nn as nn
from config import pre_prosessing_config


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        self.cnn = self._cnn_backbone()
        self.map_to_seq = nn.Linear(96, 96)
        self.rnn1 = nn.LSTM(96, 96, bidirectional=True)
        self.rnn2 = nn.LSTM(192, 96, bidirectional=True)
        self.dense = nn.Linear(192, 2)
        self.softmax = nn.Softmax(dim=1)

    def _cnn_backbone(self):
        channels = 96
        relu = nn.LeakyReLU()
        cnn_backbone = nn.Sequential()

        cnn_backbone.add_module(
            "cnn0",
            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("ReLu0", relu)
        cnn_backbone.add_module("MaxPool2d0", nn.MaxPool2d(kernel_size=(5, 1)))

        cnn_backbone.add_module(
            "cnn1",
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("ReLu1", relu)
        cnn_backbone.add_module("MaxPool2d1", nn.MaxPool2d(kernel_size=(2, 1)))

        cnn_backbone.add_module(
            "cnn2",
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("ReLu2", relu)
        cnn_backbone.add_module("MaxPool2d2", nn.MaxPool2d(kernel_size=(2, 1)))

        cnn_backbone.add_module(
            "cnn3",
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        cnn_backbone.add_module("ReLu3", relu)
        cnn_backbone.add_module("batchnorm0", nn.BatchNorm1d(3))
        cnn_backbone.add_module("MaxPool2d3", nn.MaxPool2d(kernel_size=(2, 1)))

        return cnn_backbone

    def forward(self, sample):
        sample = self.cnn(sample)
        channel, height, width = sample.size()
        sample = sample.view(channel * height, width)
        sample = sample.permute(1, 0)
        sample = self.map_to_seq(sample)
        sample, _ = self.rnn1(sample)
        sample, _ = self.rnn2(sample)
        logits = self.dense(sample)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    import torchaudio
    import torch
    from dataset import GoGoDataset
    from config import pre_prosessing_config, mel_specrogram_config

    TARGET_SAMPLE_RATE = pre_prosessing_config["target_sample_rate"]

    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=mel_specrogram_config["n_fft"],
        n_mels=mel_specrogram_config["n_mels"],
        f_max=mel_specrogram_config["f_max"],
        f_min=mel_specrogram_config["f_min"],
    )

    device = "cude" if torch.cuda.is_available() else "cpu"

    gogo = GoGoDataset("gogo-owl", TARGET_SAMPLE_RATE, transformation, device=device)
    print(gogo[0][0].shape)
    sample = gogo[0][0][:, :, :313]
    print(sample.shape)
    model = CRNN()
    output = model.forward(sample)
    print(output[0])
    pred = (output[0] == torch.max(output[0])).nonzero(as_tuple=True)[0].item()
    print(pred)
