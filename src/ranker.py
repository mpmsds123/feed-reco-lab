"""Stage 2 — Learning to Rank (LambdaMART via XGBoost rank:ndcg).

The retriever hands the ranker ~100 candidates per user; the ranker reorders them.

Training data (held-out labels):
  1. Split the training ratings 80/20 at random.
  2. Build the retriever and the rating model on the 80% only.
  3. For each user, retrieve 100 candidates from the 80% model and label each one
     with the user's held-out 20% rating: rating >= 4 -> 2, rating 3 -> 1, else 0.
     A candidate the user never rated is labelled 0.
  4. One XGBoost group per user, so the model learns ORDER within a user's list.

Why: the ranker is trained on exactly the kind of list it reorders at serve time
(retrieved, not yet rated), and the labels are real ratings the retriever never saw.
The earlier version labelled the user's top CF candidates as negatives, which are
exactly where their future likes sit, so it learned to push good items down.

Tree depth is chosen on a validation slice of the training users, never on the
test set.
"""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from eval_metrics import ndcg_at_k


def _relevance(rating: float) -> int:
    return 2 if rating >= 4 else 1 if rating >= 3 else 0


class LambdaMARTRanker:
    BASE_PARAMS = {
        "objective": "rank:ndcg", "eta": 0.1, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 2.0, "seed": 42,
    }

    def __init__(self, feature_names, depths=(2, 3, 4), rounds=200, seed=42):
        self.feature_names = feature_names
        self.depths = depths
        self.rounds = rounds
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.depth = None
        self.selection = {}

    def _train(self, groups, depth):
        X = np.vstack([F for F, _ in groups])
        y = np.concatenate([y for _, y in groups]).astype(float)
        d = xgb.DMatrix(X, label=y)
        d.set_group([len(y) for _, y in groups])
        return xgb.train({**self.BASE_PARAMS, "max_depth": depth}, d, self.rounds)

    def fit(self, groups):
        """groups: list of (feature_matrix, labels), one per user.

        Picks max_depth by mean NDCG@10 on 20% of the users, then refits on all.
        """
        idx = self.rng.permutation(len(groups))
        n_val = max(1, len(groups) // 5)
        val = [groups[i] for i in idx[:n_val]]
        tr = [groups[i] for i in idx[n_val:]]
        for depth in self.depths:
            m = self._train(tr, depth)
            self.selection[depth] = float(np.mean([
                ndcg_at_k([y[j] for j in np.argsort(-m.predict(xgb.DMatrix(F)))], 10)
                for F, y in val]))
        self.depth = max(self.selection, key=self.selection.get)
        self.model = self._train(groups, self.depth)
        return self

    def score(self, F) -> np.ndarray:
        return self.model.predict(xgb.DMatrix(F))

    def feature_importance(self) -> list[tuple[str, float]]:
        gain = self.model.get_score(importance_type="gain")      # {'f0': .., ...}
        pairs = [(self.feature_names[int(k[1:])], v) for k, v in gain.items()]
        return sorted(pairs, key=lambda x: -x[1])
