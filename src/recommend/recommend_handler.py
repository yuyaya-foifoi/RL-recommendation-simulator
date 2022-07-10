import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs.config import CFG_DICT
from src.replay_buffer.replay_buffer import ReplayBuffer


class RecommendHandler:
    def __init__(
        self,
        initial_data: tuple,
        topK: int,
        device: str,
        file_handler,
    ):

        self.ratings, self.users, self.movies, self.history = initial_data
        self.topK = topK
        self.device = device
        self.file_handler = file_handler
        self.buffer = self._get_buffer()

        self.sigmoid = nn.Sigmoid()

    def _get_buffer(self):
        buffer = ReplayBuffer(
            buffer_size=CFG_DICT["REPLAY_BUFFER"]["SIZE"],
        )
        return buffer

    def register_models(self, models):
        self.recommender, self.user_simulator = models

    def _delete_models(self):
        del self.recommender
        del self.user_simulator

    def _make_empty_pred_ratings_df(self):
        self.pred_ratings = pd.DataFrame()

    def update_ratings(self):
        self.pred_ratings = self.pred_ratings.drop("Score", axis=1)
        self.ratings = pd.concat(
            [self.ratings, self.pred_ratings]
        ).reset_index(drop=True)

    def recommend(self, time):

        if time == 0:
            self._make_empty_pred_ratings_df()
            self._initial_recommend()
            self._save_csv(time)
            self._delete_models()

        if time != 0:
            self._make_empty_pred_ratings_df()
            self._recommend()
            self._save_csv(time)
            self._delete_models()

    def _save_csv(self, time):

        self.file_handler.save(self.ratings, "ratings_{}.csv".format(time))
        self.file_handler.save(
            self.pred_ratings, "pred_ratings_{}.csv".format(time)
        )
        self.file_handler.save(self.history, "histories_{}.csv".format(time))
        self.file_handler.save(self.movies, "movies.csv")
        self.file_handler.save(self.users, "users.csv")

    def _initial_recommend(self):

        for user_idx in self.ratings.UserID.unique():
            user_idx = int(user_idx)
            single_user_feat, repeated_user_feat = self._get_user_feat(
                user_idx
            )
            movie_feat = self._get_movie_feat()
            (
                single_history_feat,
                repeated_history_feat,
            ) = self._get_history_feat(user_idx)
            pred, gt = self._initial_predict(
                repeated_user_feat, movie_feat, repeated_history_feat
            )
            topK_idxs = self._get_topK_idxs(pred)

            single_next_history_feat = single_history_feat.copy()

            for topK_idx in topK_idxs:
                single_next_history_feat[topK_idx] += 1

                state = np.hstack([single_user_feat, single_history_feat])
                next_state = np.hstack(
                    [single_user_feat, single_next_history_feat]
                )
                action = topK_idx
                reward = int(gt[action] > 0.5)

                prob = pred[topK_idx]

                self._log_rating(user_idx, action, reward, prob)
                self.buffer.add(state, action, reward, next_state, prob)

            self._log_history(user_idx, single_next_history_feat)

    def _recommend(self, topK: int = CFG_DICT["SIMULATION"]["topK"]):
        for user_idx in np.random.choice(
            self.ratings.UserID.unique(),
            size=int(len(self.ratings.UserID.unique()) / topK),
            replace=False,
        ):
            user_idx = int(user_idx)
            single_user_feat, repeated_user_feat = self._get_user_feat(
                user_idx
            )
            movie_feat = self._get_movie_feat()
            (
                single_history_feat,
                repeated_history_feat,
            ) = self._get_history_feat(user_idx)
            probs, actions, gt = self._predict(
                np.hstack([single_user_feat, single_history_feat]),
                repeated_user_feat,
                movie_feat,
                repeated_history_feat,
                topK,
            )

            single_next_history_feat = single_history_feat.copy()

            for action in actions:
                action = action.item()
                single_next_history_feat[action] += 1

                state = np.hstack([single_user_feat, single_history_feat])
                next_state = np.hstack(
                    [single_user_feat, single_next_history_feat]
                )
                reward = int(gt[action] > 0.5)
                prob = probs.squeeze()[action].item()

                self._log_rating(user_idx, action, reward, prob)
                self.buffer.add(state, action, reward, next_state, prob)

            self._log_history(user_idx, single_next_history_feat)

    def _get_timestamp(self):
        dt = datetime.datetime.utcnow()
        return int(dt.timestamp())

    def _log_rating(self, UserID, MovieID, Rating, Score):
        self.pred_ratings = self.pred_ratings.append(
            {
                "UserID": UserID,
                "MovieID": MovieID,
                "Rating": Rating,
                "Timestamp": self._get_timestamp(),
                "Score": Score,
            },
            ignore_index=True,
        )

    def _log_history(self, user_idx, history):
        self.history.loc[user_idx] = history

    def _get_user_feat(self, user_idx):
        single_user_feat = self.users.iloc[user_idx, :].to_numpy()
        repeated_user_feat = np.tile(single_user_feat, (len(self.movies), 1))
        repeated_user_feat = torch.from_numpy(repeated_user_feat).to(
            self.device
        )
        repeated_user_feat = torch.unsqueeze(repeated_user_feat, 1)
        return single_user_feat, repeated_user_feat

    def _get_movie_feat(self):
        movie_feat = torch.from_numpy(self.movies.to_numpy()).to(self.device)
        movie_feat = torch.unsqueeze(movie_feat, 1)
        return movie_feat

    def _get_history_feat(self, user_idx):
        single_history_feat = self.history.iloc[user_idx, :].to_numpy()
        repeated_history_feat = np.tile(
            single_history_feat, (len(self.movies), 1)
        )
        repeated_history_feat = torch.from_numpy(repeated_history_feat).to(
            self.device
        )
        repeated_history_feat = torch.unsqueeze(repeated_history_feat, 1)
        return single_history_feat, repeated_history_feat

    def _rank(self, arr: np.array):
        return np.argsort(np.argsort(arr))

    def _predict_initial_pred(
        self, repeated_user_feat, movie_feat, repeated_history_feat
    ):
        self.recommender.eval()
        pred = (
            (
                self.recommender(
                    repeated_user_feat, movie_feat, repeated_history_feat
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        return np.argmax(pred, axis=1)

    def _predict_gt(
        self, repeated_user_feat, movie_feat, repeated_history_feat
    ):
        self.user_simulator.eval()
        gt = (
            (
                self.user_simulator(
                    repeated_user_feat, movie_feat, repeated_history_feat
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        return np.argmax(gt, axis=1)

    def _predict_prob_action(self, single_user_feat, size: int):
        probs, action = self.recommender.get_action(
            torch.from_numpy(single_user_feat).float().to(self.device), size
        )
        return probs, action

    def _initial_predict(
        self, repeated_user_feat, movie_feat, repeated_history_feat
    ):
        pred = self._predict_initial_pred(
            repeated_user_feat, movie_feat, repeated_history_feat
        )
        gt = self._predict_gt(
            repeated_user_feat, movie_feat, repeated_history_feat
        )
        return pred, gt

    def _predict(
        self,
        single_user_feat,
        repeated_user_feat,
        movie_feat,
        repeated_history_feat,
        size,
    ):

        probs, actions = self._predict_prob_action(single_user_feat, size)
        gt = self._predict_gt(
            repeated_user_feat, movie_feat, repeated_history_feat
        )
        return probs, actions, gt

    def _get_topK_idxs(self, pred):
        is_topK = self._rank(pred) < self.topK
        topK_idxs = np.where(is_topK == True)[0]
        return topK_idxs
