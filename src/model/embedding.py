import logging

import torch
import torch.nn as nn

from configs.config import CFG_DICT

logger = logging.getLogger("src")


class UserRep(nn.Module):
    """User representation layer."""

    def __init__(self, emb_size):
        super(UserRep, self).__init__()

        self.user_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_USERS"] + 1, emb_size * 5, padding_idx=0
        )
        self.gender_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_SEX"], emb_size
        )
        self.age_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_AGES"], emb_size
        )
        self.occup_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_OCCUPS"], emb_size
        )
        self.zip_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_ZIPS"], emb_size
        )
        self.rep_dim = emb_size * 5 + emb_size * 4

    def forward(self, data):

        user_idx, sex_idx, age_idx, occupation_idx, zip_idx = self._get_data(
            data
        )

        out = torch.cat(
            [
                self.user_embedding(user_idx),
                self.gender_embedding(sex_idx),
                self.age_embedding(age_idx),
                self.occup_embedding(occupation_idx),
                self.zip_embedding(zip_idx),
            ],
            dim=1,
        )

        return out

    def _get_data(self, data):
        user_idx = data[:, 0, 0].to(torch.long)
        sex_idx = data[:, 0, 1].to(torch.long)
        age_idx = data[:, 0, 2].to(torch.long)
        occupation_idx = data[:, 0, 3].to(torch.long)
        zip_idx = data[:, 0, 4].to(torch.long)

        return user_idx, sex_idx, age_idx, occupation_idx, zip_idx


class ItemRep(nn.Module):
    """Item representation layer."""

    def __init__(self, emb_size):
        super(ItemRep, self).__init__()

        self.item_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_ITEMS"] + 1, emb_size * 5, padding_idx=0
        )
        self.year_embedding = nn.Embedding(
            CFG_DICT["DATASET"]["NUM_ITEMS"], emb_size
        )
        self.genre_linear = nn.Linear(
            CFG_DICT["DATASET"]["NUM_GENRES"], emb_size
        ).to(torch.double)
        self.rep_dim = emb_size * 5 + emb_size * 2

    def forward(self, data):

        item_idx, year_idx, genres = self._get_data(data)

        out = torch.cat(
            [
                self.item_embedding(item_idx),
                self.year_embedding(year_idx),
                self.genre_linear(genres),
            ],
            dim=1,
        )

        return out

    def _get_data(self, data):
        item_idx = data[:, 0, 0].to(torch.long)
        year_idx = data[:, 0, 1].to(torch.long)
        genres = data[:, 0, 2:].to(torch.double)
        return item_idx, year_idx, genres


class HistoryRep(nn.Module):
    """Item representation layer."""

    def __init__(self, emb_size):
        super(HistoryRep, self).__init__()
        self.history_linear = nn.Linear(
            CFG_DICT["DATASET"]["NUM_ITEMS"] + 1, emb_size * 3
        ).to(torch.double)
        self.rep_dim = emb_size * 3

    def forward(self, data):
        history = self._get_data(data)
        return self.history_linear(history)

    def _get_data(self, data):
        history = data[:, 0, 0:].to(torch.double)
        return history


class Simulator(nn.Module):
    """Simulator model that predicts the outcome of impression."""

    def __init__(self, emb_size):
        super(Simulator, self).__init__()
        self.user_rep = UserRep(emb_size)
        self.item_rep = ItemRep(emb_size)
        self.history_rep = HistoryRep(emb_size)

        self.dropout = nn.Dropout(0.1)
        self.linear = self._get_layer(emb_size)
        logger.info("Embedding size : {}".format(emb_size))
        logger.info(
            "Model Param : {}".format(self._get_model_param(self.linear))
        )

    def forward(self, user_feats, item_feats, history_feats):
        users = self.user_rep(user_feats)
        items = self.item_rep(item_feats)
        history = self.history_rep(history_feats)
        inputs = torch.cat([users, items, history], dim=1)
        return self.linear(inputs).squeeze()

    def _get_layer(self, emb_size):

        linear = nn.Sequential(
            nn.Linear(
                self.user_rep.rep_dim
                + self.item_rep.rep_dim
                + self.history_rep.rep_dim,
                emb_size * 25,
            ),
            nn.ReLU(),
            nn.Linear(emb_size * 25, emb_size * 10),
            self.dropout,
            nn.ReLU(),
            nn.Linear(emb_size * 10, emb_size * 5),
            self.dropout,
            nn.ReLU(),
            nn.Linear(emb_size * 5, emb_size * 1),
            self.dropout,
            nn.ReLU(),
            nn.Linear(emb_size * 1, 1),
            nn.ReLU(),
        ).to(torch.double)

        return linear

    def _get_model_param(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
