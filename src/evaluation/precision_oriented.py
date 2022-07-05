import numpy as np
import pandas as pd


class PrecisionOrientedMetrics:
    def __init__(self, pred_df: pd.DataFrame):
        self.pred_df = pred_df
        self.user_ids = self.pred_df.UserID.unique()
        self.n_user = len(self.user_ids)

    def _get_pred_per_user(self, user_id):
        return self.pred_df[self.pred_df.UserID == user_id].sort_values(
            "Score", ascending=False
        )

    def map(self):
        map = 0.0
        for user_id in self.user_ids:
            pred = self._get_pred_per_user(user_id)
            map += self._map_user(pred)
        return map / self.n_user

    def _map_user(self, pred):
        n_pos = 0
        precision = 0
        for i, is_item_positive in enumerate(pred.Rating, 1):
            if is_item_positive == 1:
                n_pos += 1
            precision += n_pos / i
        avg_precision = precision / (n_pos + 1e-6)
        return avg_precision

    def ndcg(self):
        ndcg = 0.0
        for user_id in self.user_ids:
            pred = self._get_pred_per_user(user_id)
            ndcg += self._ndcg_user(pred)
        return ndcg / self.n_user

    def _ndcg_user(self, pred):
        def dcg(pred):
            dcg_score = 0.0
            for i, is_item_positive in enumerate(pred.Rating):
                if is_item_positive == 1:
                    discount = np.log2(i + 2)
                    dcg_score += 1.0 / discount
            return dcg_score

        actual = dcg(pred)
        best = dcg(pred) + 1e-6
        return actual / best

    def mrr(self):
        mrr = 0.0
        for user_id in self.user_ids:
            pred = self._get_pred_per_user(user_id)
            mrr += self._mrr_user(pred)
        return mrr / self.n_user

    def _mrr_user(self, pred):
        for i, is_item_positive in enumerate(pred.Rating, 1):
            if is_item_positive == 1:
                return 1 / i
        return 0

    def hr(self):
        hr = 0.0
        for user_id in self.user_ids:
            pred = self._get_pred_per_user(user_id)
            hr += self._hr_user(pred)
        return hr / self.n_user

    def _hr_user(self, pred):
        for is_item_positive in pred.Rating:
            if is_item_positive == 1:
                return 1
        return 0
