import os
import sys
import pandas as pd
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.read import GoConfig, GoModelInfo
from src.edit_signal import *

DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

config = GoConfig()


class GoAnnotation:
    def __init__(self, path: str) -> None:
        self.path = path
        self.events = self._read_annotation()

    def _read_annotation(self):
        _, ext = os.path.splitext(self.path)
        if not os.path.isfile(self.path):
            return []
        if ext == ".csv":
            return self._read_silic_csv()
        if ext == ".txt":
            return self._read_audacity_txt()

    def _read_silic_csv(self) -> list:
        events = []
        file = pd.read_csv(self.path)
        for _, r in file.iterrows():
            events.append((r["time_begin"] / 1000, r["time_end"] / 1000))
        return events

    def _read_audacity_txt(self) -> list:
        events = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                events.append(tuple(float(l) for l in line.split("\t")[:2]))
        return events

    def arrarize(self, length: int) -> list:
        annotation = [0] * length
        for start, end in self.events:
            s = self._scale(start, length)
            e = self._scale(end, length)
            for i in range(s, e):
                annotation[i] = 1
        return torch.tensor(annotation).to(device)

    def _scale(self, time_point: float, length: int):
        return int(time_point / config.file_duration * length)


class GoSample:
    signal = None
    sample_rate = None

    def __init__(self, path: str) -> None:
        self.path = path
        self._load_audio(path)

    def _load_audio(self, path: str):
        self.signal, self.sample_rate = torchaudio.load(path)
        self.signal = self.signal.to(device)

    @property
    def spectrogram(self):
        signal = resample(self.signal, self.sample_rate, config.target_sample_rate)
        signal = mix_down(signal)
        signal = resize_duration(signal)
        signal = augment(signal.unsqueeze(1)).squeeze(1)
        spectrogram = mel(signal)
        spectrogram = am_to_db(spectrogram)
        spectrogram = normalize(spectrogram)
        return spectrogram

    @property
    def filename(self):
        return os.path.basename(self.path)


class GoDataset(Dataset):
    def __init__(
        self,
        dirname: str,
        mode: str = "train",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.dirname = dirname

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sample_path = os.path.join(self.sample_dir, self.filenames[index])
        fn = os.path.splitext(self.filenames[index])[0]
        annotation_path = self._annotation_path(fn)
        sample = GoSample(sample_path)
        annotation = GoAnnotation(annotation_path).arrarize(sample.spectrogram.shape[2])
        return sample.spectrogram, annotation

    def _annotation_path(self, filename) -> str:
        txt_path = os.path.join(self.annotation_dir, filename + ".txt")
        csv_path = os.path.join(self.annotation_dir, filename + ".csv")
        if os.path.isfile(txt_path):
            return txt_path
        return csv_path

    @property
    def filenames(self) -> list:
        filenames = []
        for filename in os.listdir(self.sample_dir):
            _, ext = os.path.splitext(filename)
            if ext.lower() == ".wav":
                filenames.append(filename)
        return filenames

    @property
    def dataset_dir(self) -> str:
        return os.path.join(DATASET_DIR, self.dirname)

    @property
    def sample_dir(self) -> str:
        return os.path.join(self.dataset_dir, self.mode, "audio")

    @property
    def annotation_dir(self) -> str:
        return os.path.join(self.dataset_dir, self.mode, "annotation")


class GoPredictSample:
    def __init__(self, path, file_duration: int, model_info: GoModelInfo) -> None:
        self.path = path
        self.file_duration = file_duration
        self.mel_specrogram = AT.MelSpectrogram(
            sample_rate=model_info.target_sample_rate,
            n_fft=model_info.n_fft,
            hop_length=model_info.hop_length,
            n_mels=model_info.n_mels,
            f_max=model_info.f_max,
            f_min=model_info.f_min,
            normalized=True,
        ).to(device)
        self._load_audio(path)

    def _load_audio(self, path: str):
        self.signal, self.sample_rate = torchaudio.load(path)
        self.signal = self.signal.to(device)

    @property
    def spectrogram(self):
        signal = resample(self.signal, self.sample_rate, config.target_sample_rate)
        signal = mix_down(signal)
        signal = resize_duration(signal, self.file_duration)
        spec = self.mel_specrogram(signal)
        spec = am_to_db(spec)
        spec = normalize(spec)
        return spec

    @property
    def filename(self):
        return os.path.basename(self.path)


class GoPredictDataset(Dataset):
    def __init__(self, path, model_info: GoModelInfo) -> None:
        self.path = path
        self.model_info = model_info
        self.audio_duration = self._max_duration()
        super().__init__()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index):
        sample_path = os.path.join(self.dirpath, self.filenames[index])
        sample = GoPredictSample(sample_path, self.audio_duration, self.model_info)
        return sample.spectrogram, sample.filename

    @property
    def isdir(self) -> bool:
        return os.path.isdir(self.path)

    @property
    def dirpath(self):
        return self.path if self.isdir else os.path.dirname(self.path)

    @property
    def filenames(self) -> list:
        filenames = []
        if self.isdir:
            for filename in os.listdir(self.path):
                _, ext = os.path.splitext(filename)
                if ext.lower() == ".wav":
                    filenames.append(filename)
            return filenames
        _, ext = os.path.splitext(self.path)
        if ext.lower() == ".wav":
            filenames.append(os.path.basename(self.path))
        return filenames

    def _max_duration(self) -> int:
        duration = 0
        for filename in self.filenames:
            path = os.path.join(self.dirpath, filename)
            info = torchaudio.info(path)
            duration = max(duration, info.num_frames / info.sample_rate)
        return int(duration + 1)


if __name__ == "__main__":
    model_info = GoModelInfo("./models/grassowl/best.pth")
    test = GoPredictDataset("./datasets/gogo-nightjar/val/audio", model_info)
    print(test.audio_duration)
    pass
