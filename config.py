pre_prosessing_config = {
    "file_duration": 3600,  # train file duration (s)
    "target_sample_rate": 24000,  # resample sample_rate
    "piece_width": 1000,  # length of each piece (number of mel hop)
    "piece_hop": 500,  # distance (mel hop) between each piece
}

mel_specrogram_config = {
    "n_fft": 512,
    "hop_length": 256,
    "n_mels": 80,
    "f_min": 1000,
    "f_max": 12000,
    # "f_max": pre_prosessing_config["target_sample_rate"] // 2,
}

training_config = {
    "epochs": 5,  # number of training epoch
    "learning_rate": 0.0005,
    "batch_size": 1,
    "dataset": "STSH_M",  # traning dataset folder name
    "model_name": "shearwater",  # folder name of training outpout
    "skip_false_rate": 0.9,  # The probability skip the training window if all annotations are 0
}
