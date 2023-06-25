import os
import sys
import torch
import torchaudio.transforms as AT
import torchvision.transforms as VT
from torch_audiomentations import Compose, Gain

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from src.read import GoConfig

DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GoConfig()

augmentation = Compose([Gain(min_gain_in_db=-10.0, max_gain_in_db=3.0)]).to(device)

mel_specrogram = AT.MelSpectrogram(
    sample_rate=config.target_sample_rate,
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    n_mels=config.n_mels,
    f_max=config.f_max,
    f_min=config.f_min,
    normalized=True,
).to(device)

am_to_db = AT.AmplitudeToDB(stype="power", top_db=80).to(device)


def resize_duration(signal, file_duration=config.file_duration):
    sr = config.target_sample_rate
    num_sample = file_duration * sr
    num_miss = max(num_sample - signal.shape[0], 0)
    signal = torch.nn.functional.pad(signal, (0, num_miss))
    signal = signal[:, :num_sample]
    return signal


def mix_down(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def resample(signal, ori_sample_rate: int, target_sample_rate):
    if ori_sample_rate != target_sample_rate:
        resampler = AT.Resample(ori_sample_rate, target_sample_rate)
        return resampler(signal)
    return signal


def augment(signal):
    return augmentation(signal, sample_rate=config.target_sample_rate)


def mel(signal):
    return mel_specrogram(signal)


def amplitude_to_db(mel_spec):
    return am_to_db(mel_spec)


def normalize(mel_spec):
    normalize = VT.Normalize(mel_spec.mean(), mel_spec.std())
    mel_spec = normalize(mel_spec)
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    return mel_spec
