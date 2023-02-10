import os
import ffmpeg
import librosa
import librosa.display
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path, PurePath
from math import ceil

FRAME_INTERVAL = 40
WINDOW_SIZE = 3000
VIDEO_EXT = ".mp4"
spec_trans = T.Spectrogram()


def make_plot_iter(spectrogram, prediction, duration_in_s):

    duration_in_ms = duration_in_s * 1000
    num_frames = int(duration_in_ms // FRAME_INTERVAL)
    max_power = spectrogram.max()
    min_power = spectrogram.min()

    pred_len_per_window = len(prediction) / num_frames
    spec_len_per_window = spectrogram.shape[1] / num_frames
    pred_window_size = round(pred_len_per_window * (WINDOW_SIZE / FRAME_INTERVAL))
    spec_window_size = round(spec_len_per_window * (WINDOW_SIZE / FRAME_INTERVAL))

    spec_full = np.full((len(spectrogram), ceil(spec_window_size / 2)), min_power)
    pred_zeros = [0] * ceil(pred_window_size // 2)

    spectrogram = np.concatenate((spec_full, spectrogram, spec_full), axis=1)
    prediction = pred_zeros + prediction + pred_zeros

    h_pred_ws = pred_window_size // 2

    pred_starts = (round(s * pred_len_per_window) for s in range(num_frames))
    pred_iter = (prediction[s : s + h_pred_ws] + [0] * h_pred_ws for s in pred_starts)
    spec_starts = (round(s * spec_len_per_window) for s in range(num_frames))
    spec_iter = (spectrogram[:, s : s + spec_window_size] for s in spec_starts)

    # test_index = -1
    # fig, axs = plt.subplots(2)
    # axs[0].plot(pred_iter[test_index])
    # axs[0].set_xlim(0, len(pred_iter[test_index]))
    # axs[0].set_ylim(-0.05, 1)
    # axs[1].imshow(spec_iter[test_index], origin="lower", aspect="auto")
    # axs[1].set_xlim(0, spec_iter[test_index].shape[1])
    # plt.show()
    # exit()

    return zip(pred_iter, spec_iter)


def save_video(ani, path):
    ff_writer = FFMpegWriter(
        fps=1000 // FRAME_INTERVAL, extra_args=["-vcodec", "libx264"]
    )
    ani.save(path, ff_writer)


def save_merge_sound(video_path, audio_path):
    video = ffmpeg.input(video_path)
    audio = ffmpeg.input(audio_path)
    video_path = "_sound".join(os.path.splitext(video_path))
    ffmpeg.concat(video, audio, v=1, a=1).output(video_path).run()
    # ffmpeg.output(video, audio, "./test.mp4").run()


def animate_mode(predictions: dict, dateset, model_info, device, source_path):
    for fn, pred in predictions.items():
        audio_path = source_path
        if os.path.isdir(source_path):
            audio_path = PurePath.joinpath(Path(source_path), Path(fn))
        signal, sr = torchaudio.load(audio_path)
        duration = signal.size()[1] / sr
        spectrogram = spec_trans(signal)
        spectrogram = librosa.power_to_db(spectrogram[0])
        plot_iter = make_plot_iter(spectrogram, pred, duration)

        fig, axs = plt.subplots(2)

        def plot_func(plot_iter):
            axs[0].clear()
            axs[1].clear()
            pred, spec = plot_iter
            num_pred = len(pred)
            probabiliy = round(pred[len(pred) // 2 - 1], 4)
            axs[0].plot(pred)
            axs[0].plot([num_pred // 2, num_pred // 2], [-0.05, 1], "r-", lw=2)
            axs[0].text(0, 1.05, f"probabiliy: {probabiliy}", color="r")
            axs[0].set_xlim(0, num_pred)
            axs[0].set_ylim(-0.05, 1)
            axs[1].imshow(spec, origin="lower", aspect="auto")
            axs[1].plot([len(spec[1]) // 2, len(spec[1]) // 2], [0, 200], "w-", lw=2)
            axs[1].set_xlim(0, spec.shape[1])

        ani = FuncAnimation(
            fig,
            plot_func,
            frames=plot_iter,
            interval=FRAME_INTERVAL,
            save_count=int(duration * 1000 // FRAME_INTERVAL),
            repeat=False,
            blit=False,
        )

        result_path = "%s%s" % (os.path.splitext(audio_path)[0], VIDEO_EXT)
        print("exporting video...")
        save_video(ani, result_path)
        save_merge_sound(result_path, audio_path)
        print("video saved to %s" % result_path)


if __name__ == "__main__":
    vp = "./data/nightjar_test/R4-GOZW_20220608_191902_demo.mp4"
    ap = "./data/nightjar_test/R4-GOZW_20220608_191902_demo.wav"
    save_merge_sound(vp, ap)
