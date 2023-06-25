import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import CRNN
from src.tool import get_device_info
from src.read import GoModelInfo
from src.data import GoPredictDataset


class GoResult:
    def __init__(
        self, model: CRNN, loader: DataLoader, model_info: GoModelInfo
    ) -> None:
        self.model = model
        self.model_info = model_info
        self._result = self._detect(loader)

    def _detect(self, loader: DataLoader) -> dict:
        results = {}
        for sample, path in tqdm(loader):
            result = self._predict(sample, self.model, path)
            results |= result
        return results

    def __str__(self) -> str:
        lines = []
        num_frame = len(next(iter(self._result.values())))
        frame_duration = self.model_info.hop_length * (
            1 / self.model_info.target_sample_rate
        )
        lines.append("Number of files: %s" % len(self._result))
        lines.append("Sample rate: %s" % self.model_info.target_sample_rate)
        lines.append("Audio duration: %f" % round(num_frame * frame_duration, 4))
        lines.append("Number of frames per result: %s" % num_frame)
        lines.append("Frame duration (s): %s" % frame_duration)
        lines.append("Minimum Frequency: %s" % self.model_info.f_min)
        lines.append("Maximum Frequency: %s" % self.model_info.f_max)
        lines.append("Filenames: %s" % list(self._result.keys()))
        return "GoResults:\n%s" % "\n".join(lines)

    @property
    def probs(self):
        return self._result

    def events(self, conf: float = 0.5, min_duration: float = 0.05) -> dict:
        """
        conf (float): confidence threshold
        min_duration (float): the duration (s) threshold of an event.

        return format:
        {
            <file name>: [
                [<start_time>, <end_time>, <average confidence>]
                ...
            ]
            ...
        }
        """
        return {
            path: self._phase_probs(a, conf, min_duration)
            for path, a in self._result.items()
        }

    def pandas(self, conf: float = 0.5, min_duration: float = 0.05) -> pd.DataFrame:
        """
        conf (float): confidence threshold
        min_duration (float): the duration (s) threshold of an event.

        dataframe field:
        | start_time | end_time | confidence | filename |
        | ---------- | -------- | ---------- | -------- |
        ...
        """

        data = []
        results = self.events(conf, min_duration)
        for path, events in results.items():
            for event in events:
                event.append(path)
            data.extend(events)

        if not data:
            return pd.DataFrame(
                [],
                columns=["start_time", "end_time", "confidence", "filename"],
            )

        dataframe = pd.DataFrame(
            np.array(data), columns=["start_time", "end_time", "confidence", "filename"]
        )
        return dataframe

    def _predict(self, sample, model, paths) -> dict:
        n = sample.shape[3]
        paths = [os.path.basename(path) for path in paths]
        pro_array = {path: [[] for _ in range(n)] for path in paths}
        start = 0

        while start + self.model_info.piece_width <= n:
            piece = sample[:, :, :, start : start + self.model_info.piece_width]
            pred = model(piece).squeeze(1)

            for i, batch in enumerate(pred):
                for j, prob in enumerate(batch):
                    pro_array[paths[i]][start + j].append(prob.item())

            start += self.model_info.piece_hop

        start = n - self.model_info.piece_width
        tail_piece = sample[:, :, :, start:n]
        pred = model(tail_piece).squeeze(1)

        for i, batch in enumerate(pred):
            for j, prob in enumerate(batch):
                pro_array[paths[i]][start + j].append(prob.item())

        pro_array = {path: self._average(a) for path, a in pro_array.items()}
        return pro_array

    def _average(self, results: list) -> list:
        return [sum(probs) / len(probs) for probs in results]

    def _phase_probs(self, probs: list, threshold=0.5, min_duration=0.05) -> list:
        s_per_frame = self.model_info.hop_length * (
            1 / self.model_info.target_sample_rate
        )

        min_num_frame = min_duration / s_per_frame
        events = []
        event_probs = []
        positive = False
        start = 0

        for i, p in enumerate(probs):
            if p <= threshold:
                if positive and i - start > min_num_frame:
                    s = round(start * s_per_frame, 4)
                    e = round(i * s_per_frame, 4)
                    ave = round(sum(event_probs) / len(event_probs), 4)
                    events.append([s, e, ave])
                event_probs = []
                positive = False
            else:
                if not positive:
                    start = i
                positive = True
                event_probs.append(p)

        return events


class GoModel:
    def __init__(self, path: str) -> None:
        self.model_info = GoModelInfo(path)
        self._device = self._device_if_available()
        self.model = self._read_model(path)

    def detect(self, path: str, bs: int = 1) -> GoResult:
        print(get_device_info(self._device))
        dataset = GoPredictDataset(path, self.model_info)
        loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False)
        return GoResult(self.model, loader, self.model_info)

    def _read_model(self, path):
        torch.cuda.empty_cache()
        model = CRNN()
        model.load_state_dict(torch.load(path, map_location=torch.device(self._device)))
        model.to(self._device)
        model.eval()
        return model

    @property
    def device(self):
        return self._device

    def cpu(self):
        self._device = "cpu"
        self.model.to(self._device)

    def cuda(self):
        self._device = self._device_if_available()
        self.model.to(self._device)

    def _device_if_available(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
