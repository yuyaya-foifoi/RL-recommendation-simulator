import numpy as np
import pandas as pd


class PrecisionOrientedMetrics:
    def __init__(self, pred_df: pd.DataFrame, topK):
        self.pred_df = pred_df
        self.user_ids = self.pred_df.UserID.unique()
        self.n_user = len(self.user_ids)
        self.topK = topK

        self.UserID2ratings = self._get_UserID2ratings()

    def _get_UserID2ratings(self):
        return (
            self.pred_df.groupby("UserID")
            .apply(
                lambda x: x.sort_values(
                    "Score", ascending=False
                ).Rating.to_list()
            )
            .to_dict()
        )

    def _precision_at_k(self, ratings, atK):
        ratings = ratings[:atK]
        return np.sum(ratings) / atK

    def _average_precision(self, ratings):
        ap = 0.0
        for atK, is_item_positive in enumerate(ratings, 1):
            if is_item_positive > 0:
                ap += self._precision_at_k(ratings, atK)
        if np.sum(ratings) > 0:
            return ap / np.sum(ratings)
        else:
            return 0.0

    def mean_Precision(self):
        mean_p = 0.0
        for user_id in self.user_ids:
            rating = self.UserID2ratings[user_id]
            mean_p += self._precision_at_k(rating, self.topK)
        return mean_p / self.n_user

    def MAP(self):
        map = 0.0
        for user_id in self.user_ids:
            rating = self.UserID2ratings[user_id]
            map += self._average_precision(rating)
        return map / self.n_user

    def ndcg(self):
        ndcg = 0.0
        for user_id in self.user_ids:
            ratings = self.UserID2ratings[user_id]
            ndcg += self._ndcg_user(ratings)
        return ndcg / self.n_user

    def _ndcg_user(self, ratings):
        def dcg(ratings):
            dcg_score = 0.0
            for i, is_item_positive in enumerate(ratings):
                if is_item_positive == 1:
                    discount = np.log2(i + 2)
                    dcg_score += 1.0 / discount
            return dcg_score

        actual = dcg(ratings)
        best = dcg(ratings) + 1e-6
        return actual / best

    def mrr(self):
        mrr = 0.0
        for user_id in self.user_ids:
            ratings = self.UserID2ratings[user_id]
            mrr += self._mrr_user(ratings)
        return mrr / self.n_user

    def _mrr_user(self, ratings):
        for i, is_item_positive in enumerate(ratings, 1):
            if is_item_positive == 1:
                return 1 / i
        return 0

    def hr(self):
        hr = 0.0
        for user_id in self.user_ids:
            ratings = self.UserID2ratings[user_id]
            hr += self._hr_user(ratings)
        return hr / self.n_user

    def _hr_user(self, ratings):
        for is_item_positive in ratings:
            if is_item_positive == 1:
                return 1
        return 0
