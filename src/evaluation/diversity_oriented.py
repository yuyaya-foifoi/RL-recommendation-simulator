import itertools

import numpy as np
import pandas as pd


class DiversityOrientedMetrics:
    def __init__(
        self, prediction: pd.DataFrame, history: pd.DataFrame, topK: int = 5
    ) -> None:

        self.topK = topK
        self.prediction = prediction
        self.history = history

        self.unique_user_ids = np.sort(prediction.UserID.unique().tolist())
        self.unique_MovieIDs = (
            prediction.MovieID.unique().tolist()
            + history.MovieID.unique().tolist()
        )

        self.prefs_dict = self._get_prefs_dict()

    def _prefs(self, MovieID: int) -> int:
        return len(self.history[self.history.MovieID == MovieID])

    def _prefs_both(self, MovieID1: int, MovieID2: int) -> int:
        recommended_both = self.history[
            (self.history.MovieID == MovieID1)
            | (self.history.MovieID == MovieID2)
        ]
        series_user = recommended_both.MovieID.value_counts()
        return series_user[series_user == 2].count()

    def _get_prefs_dict(self) -> dict:
        prefs_dict = {}
        for MovieID in self.unique_MovieIDs:
            prefs_dict[MovieID] = self._prefs(MovieID)
        return prefs_dict

    def _diversity_user(self, user_id):
        diversity_score = 0.0

        rec_ID_per_user = self.prediction[
            self.prediction.UserID == user_id
        ].MovieID

        for i, j in list(itertools.combinations(rec_ID_per_user, 2)):
            pref_both = self._prefs_both(i, j)
            if pref_both != 0:
                diversity_score += (
                    np.sqrt(self.prefs_dict[i])
                    * np.sqrt(self.prefs_dict[j])
                    / pref_both
                )
        return diversity_score

    def diversity_score(self) -> float:
        diversity_score = 0.0

        for user_id in self.unique_user_ids:
            diversity_score += self._diversity_user(user_id)

        return diversity_score / len(self.unique_user_ids)

    def _serendipity_user(self, user_id):
        serendipity_score = 0.0

        rec_ID_per_user = self.prediction[
            self.prediction.UserID == user_id
        ].MovieID
        con_ID_per_user = self.history[self.history.UserID == user_id].MovieID

        for rec_item, con_item in list(
            itertools.product(rec_ID_per_user, con_ID_per_user)
        ):
            pref_both = self._prefs_both(rec_item, con_item)
            if pref_both != 0:
                serendipity_score += (
                    np.sqrt(self.prefs_dict[rec_item])
                    * np.sqrt(self.prefs_dict[con_item])
                    / pref_both
                )
        if len(con_ID_per_user) > 0:
            return serendipity_score / len(con_ID_per_user)
        else:
            return 0.0

    def serendipity_score(self):
        serendipity_score = 0.0

        for user_id in self.unique_user_ids:
            serendipity_score += self._serendipity_user(user_id)

        return serendipity_score / len(self.unique_user_ids)

    def _novelty_user(self, user_id):
        novelty_score = 0.0

        rec_fkus_per_user = self.prediction[
            self.prediction.UserID == user_id
        ].MovieID

        for rec_fku in rec_fkus_per_user:
            prefs = self._prefs(rec_fku)
            if prefs != 0:
                novelty_score += (
                    np.log2(len(self.unique_user_ids) / prefs) / self.topK
                )
        return novelty_score

    def novelty_score(self) -> float:
        novelty_score = 0.0
        for user_id in self.unique_user_ids:
            novelty_score += self._novelty_user(user_id)
        return novelty_score / len(self.unique_user_ids)

    def uniqueness_score(self) -> float:
        return len(self.prediction.MovieID.unique()) / self.topK
