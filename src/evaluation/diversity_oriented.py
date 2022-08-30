import collections
import itertools

import numpy as np
import pandas as pd


class DiversityOrientedMetrics:
    def __init__(
        self, prediction: pd.DataFrame, ratings: pd.DataFrame, topK: int = 5
    ) -> None:

        self.topK = topK
        self.prediction = prediction
        self.ratings = ratings

        self.unique_user_ids = np.sort(prediction.UserID.unique().tolist())
        self.unique_MovieIDs = (
            prediction.MovieID.unique().tolist()
            + ratings.MovieID.unique().tolist()
        )

        self.MovieID2UserIDs = (
            self.ratings.groupby("MovieID")
            .apply(lambda x: x.UserID.to_list())
            .to_dict()
        )
        self.UserID2rating_MovieIDs = (
            self.ratings.groupby("UserID")
            .apply(lambda x: x.MovieID.to_list())
            .to_dict()
        )
        self.UserID2prediction_MovieIDs = (
            self.prediction.groupby("UserID")
            .apply(lambda x: x.MovieID.to_list())
            .to_dict()
        )

    def _get_movieid_list_from_userid(self, userid2movieid, UserID):
        if UserID in userid2movieid.keys():
            return list(userid2movieid[UserID])
        else:
            return []

    def _get_userid_list_from_movieid(self, MovieID):
        if MovieID in self.MovieID2UserIDs.keys():
            return list(self.MovieID2UserIDs[MovieID])
        else:
            return []

    def _prefs(self, MovieID: int) -> int:
        return len(self._get_userid_list_from_movieid(MovieID))

    def _prefs_both(self, MovieID1: int, MovieID2: int) -> int:
        user_id_set1 = self._get_userid_list_from_movieid(MovieID1)
        user_id_set2 = self._get_userid_list_from_movieid(MovieID2)

        if user_id_set1 == user_id_set2:
            counter = collections.Counter(user_id_set1)
        else:
            counter = collections.Counter(user_id_set1 + user_id_set2)

        return np.sum(np.array(list(counter.values())) == 2)

    def _diversity_user(self, user_id):
        diversity_score = 0.0

        rec_ID_per_user = self._get_movieid_list_from_userid(
            self.UserID2prediction_MovieIDs, user_id
        )

        for i, j in list(itertools.combinations(rec_ID_per_user, 2)):
            pref_both = self._prefs_both(i, j)
            if pref_both != 0:
                diversity_score += (
                    np.sqrt(self._prefs(i))
                    * np.sqrt(self._prefs(j))
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

        rec_ID_per_user = self._get_movieid_list_from_userid(
            self.UserID2prediction_MovieIDs, user_id
        )
        con_ID_per_user = self._get_movieid_list_from_userid(
            self.UserID2rating_MovieIDs, user_id
        )

        for rec_item, con_item in list(
            itertools.product(rec_ID_per_user, con_ID_per_user)
        ):
            pref_both = self._prefs_both(rec_item, con_item)
            if pref_both != 0:
                serendipity_score += (
                    np.sqrt(self._prefs(rec_item))
                    * np.sqrt(self._prefs(con_item))
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

        rec_ID_per_user = self._get_movieid_list_from_userid(
            self.UserID2prediction_MovieIDs, user_id
        )

        for rec_fku in rec_ID_per_user:
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
