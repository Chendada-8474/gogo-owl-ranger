import os
import torch
import torchaudio
import torchaudio.transforms as AT
import torchvision.transforms as VT
import pandas as pd
from pathlib import Path, PurePath
from torch.utils.data import Dataset
from torch_audiomentations import Compose, Gain
from utils.config import mel_specrogram_config
from math import ceil

DATASETS_DIR = PurePath.joinpath(Path(__file__).parent.parent, Path("datasets"))
ANNOTATION_EXTENSIONS = (".txt", ".csv")


class GoGoDataset(Dataset):
    def __init__(
        self,
        dataset_dirname: str,
        target_sample_rate: int,
        mode: str = "train",
        device: str = "cpu",
    ):
        assert (
            mode == "train" or mode == "val" or mode == "predict"
        ), "only train, val or predict mode"

        self.target_sample_rate = target_sample_rate
        self.mode = mode
        self.target_dataset_dir = self._get_target_dataset_dir(
            dataset_dirname, self.mode
        )

        self.sample_names = self._all_sample_filename()
        self.annotations = self._all_annotations()
        self.device = device
        self.augmentation = Compose(
            transforms=[
                Gain(
                    min_gain_in_db=-10.0,
                    max_gain_in_db=3.0,
                    p=0.5,
                )
            ]
        )
        self.mel_specrogram = AT.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=mel_specrogram_config["n_fft"],
            n_mels=mel_specrogram_config["n_mels"],
            f_max=mel_specrogram_config["f_max"],
            f_min=mel_specrogram_config["f_min"],
            normalized=True,
        ).to(self.device)

        self.amplitude_to_db = AT.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):

        audio_path = self._get_audio_sample_path(index)

        try:
            signal, sample_rate = torchaudio.load(audio_path)
            signal = signal.to(self.device)
        except IOError:
            print("Corrupted audio for %d" % index)
            return self[index + 1]

        ori_num_sample = signal.shape[1]
        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self._mix_down_if_necessary(signal)
        signal = self.augmentation(signal.unsqueeze(1)).squeeze(1)
        signal = self.mel_specrogram(signal)
        signal = self.amplitude_to_db(signal)
        normalize = VT.Normalize(signal.mean(), signal.std())
        signal = normalize(signal)
        signal -= signal.min()
        signal /= signal.max()
        spectrogram_width = signal.shape[2]

        label = self._generate_annotation(index, spectrogram_width, ori_num_sample)

        return signal, torch.tensor(label)

    def _get_target_dataset_dir(self, dataset_dirname, mode):
        return Path.joinpath(DATASETS_DIR, Path(dataset_dirname), Path(mode))

    def _all_sample_filename(self) -> list:
        dir_path = Path.joinpath(self.target_dataset_dir, Path("audio"))
        files = [file for file in os.listdir(dir_path) if file.lower().endswith(".wav")]
        return files

    def _all_annotations(self) -> set:
        dir_path = Path.joinpath(self.target_dataset_dir, Path("annotation"))
        return set(os.listdir(dir_path))

    def _get_audio_sample_path(self, index):
        dir_path = Path.joinpath(self.target_dataset_dir, Path("audio"))
        return Path.joinpath(dir_path, Path(self.sample_names[index]))

    def _get_annotation_path(self, txt_filename):
        dir_path = Path.joinpath(self.target_dataset_dir, Path("annotation"))
        return Path.joinpath(dir_path, Path(txt_filename))

    def _resample_if_necessary(self, signal, sample_rate):
        assert sample_rate >= self.target_sample_rate, (
            """
        The sample rate of signal has to be higher than %s kHz
        """
            % self.target_sample_rate
            // 1000
        )

        if sample_rate > self.target_sample_rate:
            resampler = AT.Resample(
                sample_rate,
                self.target_sample_rate,
            )
            signal = resampler(signal)

        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_dataset_root_dir(self):
        return PurePath.joinpath(Path(__file__).parent.parent, Path("datasets"))

    def _generate_annotation(self, index, spectrogram_width, ori_num_sample):
        annotation_filename = os.path.splitext(self.sample_names[index])[0]
        annotation_ext_fns = [
            "%s%s" % (annotation_filename, ext) for ext in ANNOTATION_EXTENSIONS
        ]

        new_annotation = [0] * spectrogram_width

        for ext_fn in annotation_ext_fns:
            if ext_fn not in self.annotations:
                continue

            annotation_path = self._get_annotation_path(ext_fn)
            time_ranges = self._read_annotation(annotation_path)

            for s, e in time_ranges:
                s = self._scale_range_for_spec(s, spectrogram_width, ori_num_sample)
                e = self._scale_range_for_spec(e, spectrogram_width, ori_num_sample)
                for time_point in range(int(s), ceil(e)):
                    new_annotation[time_point] = 1

        return new_annotation

    def _scale_range_for_spec(self, num, spectrogram_width, ori_num_sample):
        num = (num / (ori_num_sample / self.target_sample_rate)) * spectrogram_width
        return num

    def _read_annotation(self, annotation_path):
        time_ranges = []
        file_name, extension = os.path.splitext(annotation_path)

        if extension.lower() == ".txt":
            with open(annotation_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    time_ranges.append([float(l) for l in line.split("\t")[:2]])

        elif extension.lower() == ".csv":
            csv_anno = pd.read_csv(annotation_path)
            for time_begin, time_end in zip(
                csv_anno["time_begin"], csv_anno["time_end"]
            ):
                time_ranges.append([time_begin / 1000, time_end / 1000])

        return time_ranges


class PredictDataset(Dataset):
    def __init__(self, source_path, model_info: dict, device="cpu") -> None:
        self.device = device
        self.source_path = source_path
        self.target_sample_rate = model_info["target_sample_rate"]
        self.audio_paths = self._all_audio_paths(self.source_path)

        self.mel_specrogram = AT.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=model_info["n_fft"],
            n_mels=model_info["n_mels"],
            f_max=model_info["f_max"],
            f_min=model_info["f_min"],
            normalized=True,
        ).to(self.device)

        self.amplitude_to_db = AT.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        try:
            signal, sample_rate = torchaudio.load(self.audio_paths[index])
            signal = signal.to(self.device)
        except IOError:
            print("Corrupted audio for %d" % index)
            return self[index + 1]
        signal = self.mel_specrogram(signal)
        signal = self.amplitude_to_db(signal)
        normalize = VT.Normalize(signal.mean(), signal.std())
        signal = normalize(signal)
        signal -= signal.min()
        signal /= signal.max()
        return signal, str(self.audio_paths[index])

    def _all_audio_paths(self, source_path) -> list:
        source_path = Path(source_path)
        if source_path.is_dir():
            return [
                PurePath.joinpath(source_path, fn)
                for fn in os.listdir(source_path)
                if fn.lower().endswith(".wav")
            ]
        return [source_path]


if __name__ == "__main__":
    pass
