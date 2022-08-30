from torch.utils.data import DataLoader

from configs.config import CFG_DICT
from src.dataloader.movielens import MovieLensDataset


def get_MovieLensDataloaders(model_type: str, processed_data, data_split: str):

    train_loader = DataLoader(
        MovieLensDataset(processed_data, data_type="train"),
        batch_size=CFG_DICT[model_type]["BS"]["train"],
        shuffle=True,
    )

    val_loader = DataLoader(
        MovieLensDataset(processed_data, data_type="val"),
        batch_size=CFG_DICT[model_type]["BS"]["val"],
        shuffle=False,
    )

    test_loader = DataLoader(
        MovieLensDataset(processed_data, data_type="test"),
        batch_size=CFG_DICT[model_type]["BS"]["test"],
        shuffle=False,
    )

    return (train_loader, val_loader, test_loader)
