import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, datasets, data_type="train", data_split="random"):

        self.ratings, self.users, self.movies, self.history = datasets
        self.train, self.val, self.test = self._split(data_split)
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
        score = np.array(series.Rating, dtype=int)

        return (
            torch.from_numpy(user_feat),
            torch.from_numpy(item_feat),
            torch.from_numpy(history_feat),
            torch.from_numpy(score),
        )

    def _split(self, data_split):
        """
        split by ratio manner with time-aware
        """
        if data_split == "random":
            train, val, test = self._random_split(self.ratings)
            return train, val, test

        elif data_split == "time_aware":
            train, val, test = self._time_aware_split(self.ratings)
            return train, val, test

        else:
            raise NotImplementedError()

    def _random_split(self, ratings: pd.DataFrame):
        TEST_SIZE = 0.1
        random_state = np.random.RandomState(42)

        permuted_index = random_state.permutation(len(ratings.index))

        train_idxs, val_idxs, test_idxs = np.split(
            permuted_index,
            [
                int(len(permuted_index) * (1 - TEST_SIZE * 2)),
                int(len(permuted_index) * (1 - TEST_SIZE)),
            ],
        )

        train = ratings.loc[ratings.index.isin(train_idxs)]
        val = ratings.loc[ratings.index.isin(val_idxs)]
        test = ratings.loc[ratings.index.isin(test_idxs)]

        return train, val, test

    def _time_aware_split(self, ratings: pd.DataFrame):
        val_from = ratings.Timestamp.quantile(0.8)
        test_from = ratings.Timestamp.quantile(0.9)

        train = ratings[(val_from > ratings.Timestamp)]
        val = ratings[
            ((val_from <= ratings.Timestamp) & (ratings.Timestamp < test_from))
        ]
        test = ratings[(ratings.Timestamp >= test_from)]
        return train, val, test

    def _select_data(self, data_type: str):
        if data_type == "train":
            return self.train

        elif data_type == "val":
            return self.val

        elif data_type == "test":
            return self.test

        else:
            raise NotImplementedError
