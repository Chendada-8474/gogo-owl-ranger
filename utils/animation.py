import os
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path, PurePath


def animate_mode(predictions: dict, dateset, model_info, device):
    for spctrogram, path in dateset:
        fn = os.path.basename(path)
        pass
    for fn, pred in predictions.items():
        print(fn)
        # audio_path = source_path
        # if os.path.isdir(source_path):
        #     audio_path = PurePath.joinpath(Path(source_path), Path(fn))

    pass
