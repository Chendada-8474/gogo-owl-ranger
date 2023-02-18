from os import cpu_count

pre_prosessing_config = {
    "target_sample_rate": 16000,  # resample sample_rate
    "sample_duration": 5,  # length (s) of each sample
    "sample_hop": 1,  # distance between each sample
}

mel_specrogram_config = {
    "n_fft": 512,
    "n_mels": 64,
    "f_min": 1000,
    # "f_max": pre_prosessing_config["target_sample_rate"] // 2,
    "f_max": 4000,
}

training_config = {
    "epochs": 15,  # number of training epoch
    "learning_rate": 0.0005,
    "batch_size": 64,
    "dataset": "gogo-nightjar",  # traning dataset folder name
    "cpu_workers": cpu_count(),
    "skip_false_rate": 0.8,  # The probability skip the training window if all annotations are 0
    "model_name": "nightjar",  # folder name of training outpout
}
