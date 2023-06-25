import torch
from torch.utils.data import DataLoader
from config import *
from src.model import CRNN
from src.data import GoDataset
from src.trainer import GoTrain
from src.tool import get_device_info
from src.read import GoConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GoConfig()


def main():
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(get_device_info(device))

    train_set = GoDataset(config.dataset, mode="train")
    val_set = GoDataset(config.dataset, mode="val")
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = CRNN().to(device)
    loss_func = torch.nn.BCELoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    go_model = GoTrain(train_loader, val_loader, model, loss_func, optimiser)
    go_model.tune()
    go_model.save()


if __name__ == "__main__":
    main()
