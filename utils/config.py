from os import cpu_count

pre_prosessing_config = {
    "target_sample_rate": 16000,  # resample sample_rate
    "sample_duration": 5,  # length (s) of each sample
    "sample_hop": 1,  # distance between each sample
}

mel_specrogram_config = {
    "n_fft": 256,
    "n_mels": 64,
    "f_min": 1000,
    # "f_max": pre_prosessing_config["target_sample_rate"] // 2,
    "f_max": 4000,
}

training_config = {
    "epochs": 4,  # number of training epoch
    "learning_rate": 0.001,
    "batch_size": 2,
    "dataset": "gogo-nightjar",  # traning dataset folder name
    "cpu_workers": cpu_count(),
    "skip_false_rate": 0.6,  # The probability skip the training window if all annotations are 0
    "model_name": "nightjar",  # folder name of training outpout
}
