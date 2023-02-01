import os
import torch
import torchaudio
from math import ceil
from pathlib import Path, PurePath
from torch.utils.data import Dataset
from utils.config import mel_specrogram_config

DATASETS_DIR = PurePath.joinpath(Path(__file__).parent.parent, Path("datasets"))


class GoGoDataset(Dataset):
    def __init__(
        self,
        dataset_dirname: str,
        target_sample_rate: int,
        transform,
        mode: str = "train",
        device: str = "cpu",
    ):
        assert mode == "train" or mode == "val", "only train or val mode"

        self.target_sample_rate = target_sample_rate
        self.mode = mode
        self.target_dataset_dir = self._get_target_dataset_dir(
            dataset_dirname, self.mode
        )
        self.sample_names = self._all_sample_filename()
        self.annotations = self._all_annotations()
        self.device = device
        self.transform = transform.to(self.device)

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
        signal = self.transform(signal)
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
            resampler = torchaudio.transforms.Resample(
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
        annotation_filename = "%s.%s" % (
            os.path.splitext(self.sample_names[index])[0],
            "txt",
        )

        new_annotation = [0] * spectrogram_width

        if annotation_filename not in self.annotations:
            return new_annotation

        annotation_path = self._get_annotation_path(annotation_filename)
        time_ranges = self._read_annotation_txt(annotation_path)

        for s, e in time_ranges:
            s = self._scale_range_for_spec(s, spectrogram_width, ori_num_sample)
            e = self._scale_range_for_spec(e, spectrogram_width, ori_num_sample)
            for time_point in range(int(s), ceil(e) + 1):
                new_annotation[time_point] = 1
        return new_annotation

    def _scale_range_for_spec(self, num, spectrogram_width, ori_num_sample):
        num = (num / (ori_num_sample / self.target_sample_rate)) * spectrogram_width
        return num

    def _read_annotation_txt(self, txt_path):
        time_ranges = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                time_ranges.append([float(l) for l in line.split("\t")[:2]])
        return time_ranges


if __name__ == "__main__":
    from config import pre_prosessing_config

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
    print(gogo[0])
