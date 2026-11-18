"""Star-rating model ported from the DSCI 553 hybrid recommender (competition.py).

Predicts the stars a user would give a business from the 28 Yelp features:
    prediction = 0.65 * XGBoost + 0.35 * sklearn GradientBoosting, clipped to [1, 5]
Hyperparameters are copied unchanged from competition.py (validation RMSE 0.9450 there).

The original also blends in an item-item Pearson CF estimate with weight 0.01-0.15.
That part is left out here: computing Pearson neighbours for ~1M (user, candidate)
pairs in plain Python is too slow, and it carries at most 15% of the original blend.

In this repo the predicted rating is one input to the Stage 2 ranker.
"""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor


class HybridRatingModel:
    def __init__(self, feature_store):
        self.fs = feature_store
        self.xgb = xgb.XGBRegressor(
            n_estimators=250, max_depth=8, learning_rate=0.050, random_state=42,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
            reg_alpha=0.05, reg_lambda=0.9, n_jobs=-1)
        self.gb = GradientBoostingRegressor(
            n_estimators=230, max_depth=7, learning_rate=0.065, random_state=42,
            subsample=0.85, min_samples_split=20, min_samples_leaf=10, max_features="sqrt")

    def fit(self, rows):
        """rows: [(user_id, business_id, stars)]"""
        X = np.asarray([self.fs.build(u, b) for u, b, _ in rows], dtype=float)
        y = np.asarray([r for _, _, r in rows], dtype=float)
        self.xgb.fit(X, y)
        self.gb.fit(X, y)
        return self

    def predict(self, user_id, item_ids) -> np.ndarray:
        X = np.asarray([self.fs.build(user_id, b) for b in item_ids], dtype=float)
        pred = 0.65 * self.xgb.predict(X) + 0.35 * self.gb.predict(X)
        return np.clip(pred, 1.0, 5.0)
