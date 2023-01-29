from os import cpu_count

pre_prosessing_config = {
    "target_sample_rate": 16000,
    "sample_duration": 5,  # length (s) of each sample
    "sample_hop": 1,  # distance between each sample
}

mel_specrogram_config = {
    "n_fft": 512,
    "n_mels": 64,
    "f_min": 1,
    "f_max": pre_prosessing_config["target_sample_rate"] // 2,
}

training_config = {
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 8,
    "dataset": "gogo-owl",
    "val_proportion": 0.1,
    "cpu_workers": cpu_count() // 2,
}
