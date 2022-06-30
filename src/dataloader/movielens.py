import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, datasets, data_type="train"):

        self.ratings, self.users, self.movies, self.history = datasets
        self.train, self.val, self.test = self._split()
        self.data = self._select_data(data_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data.iloc[idx, :]

        user_feat = self.users[self.users.UserID == series.UserID].to_numpy()
        item_feat = self.movies[
            self.movies.MovieID == series.MovieID
        ].to_numpy()
        history_feat = self.history[
            self.history.index == series.UserID
        ].to_numpy()
        score = np.array(series.Rating)

        return (
            torch.from_numpy(user_feat),
            torch.from_numpy(item_feat),
            torch.from_numpy(history_feat),
            torch.from_numpy(score),
        )

    def _split(self):
        """
        split by ratio manner with time-aware
        """
        _train, test = self._time_aware_split(self.ratings, quantile=0.9)
        train, val = self._time_aware_split(_train, quantile=0.9)
        return train, val, test

    def _time_aware_split(self, ratings: pd.DataFrame, quantile: float):
        thresh_timestamp = ratings.Timestamp.quantile(quantile)
        majority = ratings[(thresh_timestamp > ratings.Timestamp)]
        minority = ratings[~(thresh_timestamp > ratings.Timestamp)]
        return majority, minority

    def _select_data(self, data_type: str):
        if data_type == "train":
            return self.train

        elif data_type == "val":
            return self.val

        elif data_type == "test":
            return self.test

        else:
            raise NotImplementedError
