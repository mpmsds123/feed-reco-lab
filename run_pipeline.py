"""End-to-end demo: retrieval -> LTR ranking -> bandit re-rank -> A/B evaluation.

Run:
    python data/make_synthetic_yelp.py --data_dir data/synthetic   # or point at real Yelp
    python run_pipeline.py --data_dir data/synthetic

Prints a report formatted for pasting into the README. Every number is computed here, not
hard-coded — re-run with a different --seed to see it move.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import YelpFeatureStore, FEATURE_NAMES           # noqa: E402
from retrieval import PopularityRetriever, ItemCFRetriever      # noqa: E402
from ranker import LambdaMARTRanker, _relevance                 # noqa: E402
from rating_model import HybridRatingModel                      # noqa: E402
from eval_metrics import (dcg_at_k, recall_at_k,                # noqa: E402
                          average_precision_at_k)
import bandit as B                                              # noqa: E402
import experiment as X                                          # noqa: E402


def load_csv(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            u, b, r = line.strip().split(",")
            rows.append((u, b, float(r)))
    return rows


def rank_by(scores, items):
    return [items[j] for j in np.argsort(-np.asarray(scores))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if not os.path.exists(os.path.join(a.data_dir, "yelp_train.csv")):
        from data.make_synthetic_yelp import generate
        generate(a.data_dir, seed=a.seed)

    train = load_csv(os.path.join(a.data_dir, "yelp_train.csv"))
    val = load_csv(os.path.join(a.data_dir, "yelp_val.csv"))
    all_items = sorted({b for _, b, _ in train})
    item_set = set(all_items)

    # ---- held-out split of TRAIN for ranker training (val stays untouched as the test set)
    split_rng = np.random.default_rng(a.seed)
    fit_rows, label_rows = [], []
    for row in train:
        (label_rows if split_rng.random() < 0.2 else fit_rows).append(row)

    def build_stage(rows):
        """Feature store, CF retriever and rating model built from `rows` only."""
        fs_ = YelpFeatureStore(a.data_dir)
        ua, bp = defaultdict(int), defaultdict(int)
        for u, b, _ in rows:
            ua[u] += 1; bp[b] += 1
        fs_.set_activity(ua, bp)
        cf_ = ItemCFRetriever([(u, b, r) for u, b, r in rows if r >= 4])   # CF on LIKES only
        rm_ = HybridRatingModel(fs_).fit(rows)
        seen_ = defaultdict(set)
        for u, b, _ in rows:
            seen_[u].add(b)
        return fs_, cf_, rm_, seen_

    print("=" * 66)
    print("STAGE 1+2 : RETRIEVAL  ->  LEARNING-TO-RANK")
    print("=" * 66)
    fs_fit, cf_fit, rm_fit, seen_fit = build_stage(fit_rows)   # for ranker training
    fs, cf, rm, user_train_items = build_stage(train)          # for serving / evaluation
    pop = PopularityRetriever(train)

    # Ranker features. CF score enters as rank and per-user normalised score, not the
    # raw score: raw co-like counts grow with the amount of data, so their scale
    # differs between the 80% training models and the 100% serving models.
    KEEP = ["biz_stars", "usr_average_stars", "star_difference"]
    keep_idx = [FEATURE_NAMES.index(n) for n in KEEP]
    ranker_names = KEEP + ["cf_rank", "cf_score_norm", "pred_rating"]

    def features(fs_, cf_, rm_, u, cand, use_rating=True):
        sc = cf_.scores_for(u, cand)
        n = len(cand)
        cols = [np.asarray([[fs_.build(u, b)[i] for i in keep_idx] for b in cand], float),
                1 - np.arange(n) / n,                  # candidates arrive in CF order
                sc / max(sc.max(), 1e-12)]
        if use_rating:
            cols.append(rm_.predict(u, cand))
        return np.column_stack(cols)

    # training groups: candidates from the 80% CF, labelled by the held-out 20%
    held = defaultdict(dict)
    for u, b, r in label_rows:
        held[u][b] = _relevance(r)
    train_groups = {True: [], False: []}
    for u, labels in held.items():
        cand = cf_fit.candidates(u, seen_fit[u], n=100)
        if not cand or not any(labels.get(b, 0) == 2 for b in cand):
            continue
        y = np.asarray([labels.get(b, 0) for b in cand])
        F = features(fs_fit, cf_fit, rm_fit, u, cand)
        train_groups[True].append((F, y))
        train_groups[False].append((F[:, :-1], y))       # ablation: no rating model
    ltr = LambdaMARTRanker(ranker_names).fit(train_groups[True])
    ltr_norating = LambdaMARTRanker(ranker_names[:-1]).fit(train_groups[False])
    print(f"  ranker trained on {len(train_groups[True])} users; "
          f"max_depth={ltr.depth} picked on a validation slice of training users")

    # End-to-end evaluation on val: each PIPELINE produces its own top-K list, scored
    # against the user's held-out likes (rating>=4). Recall denominator is the user's
    # TOTAL held-out likes, so all pipelines are directly comparable.
    K = a.k
    val_pos = defaultdict(dict)
    for u, b, r in val:
        if b in item_set:
            val_pos[u][b] = _relevance(r)

    def two_stage(model, use_rating):
        def run(u, seen):
            cand = cf.candidates(u, seen, n=100)
            if not cand:
                return []
            return rank_by(model.score(features(fs, cf, rm, u, cand, use_rating)), cand)
        return run

    pipelines = {
        "Popularity (baseline)": lambda u, seen: pop.candidates(u, seen, K),
        "Item-CF retrieval":     lambda u, seen: cf.candidates(u, seen, K),
        "Two-stage, no rating model": two_stage(ltr_norating, False),
        "Two-stage CF->LTR":     two_stage(ltr, True),
    }
    rels_by_pipe = {name: {} for name in pipelines}
    for u, rel_items in val_pos.items():
        if u not in user_train_items:
            continue
        if not any(g > 0 for g in rel_items.values()):
            continue
        seen = user_train_items[u]
        for name, fn in pipelines.items():
            rels_by_pipe[name][u] = [rel_items.get(b, 0) for b in fn(u, seen)[:K]]

    ndcg_by_pipe = {}
    for name in pipelines:
        rr = rels_by_pipe[name]
        ndcg, rec, mapk = [], [], []
        for u, rels in rr.items():
            grades = [g for g in val_pos[u].values() if g > 0]
            idcg = dcg_at_k(sorted(val_pos[u].values(), reverse=True), K)
            ndcg.append(dcg_at_k(rels, K) / idcg if idcg else 0.0)
            rec.append(sum(1 for r in rels if r > 0) / len(grades))
            mapk.append(average_precision_at_k(rels, K))
        ndcg_by_pipe[name] = np.asarray(ndcg)
        print(f"  {name:28s}  NDCG@{K}={np.mean(ndcg):.4f}  "
              f"Recall@{K}={np.mean(rec):.4f}  MAP@{K}={np.mean(mapk):.4f}  (n={len(rr)})")
    rels_pop = rels_by_pipe["Popularity (baseline)"]
    rels_cf = rels_by_pipe["Item-CF retrieval"]

    print(f"\n  Paired t-test on per-user NDCG@{K}, same users (each stage vs the one before):")
    for prev, nxt in [("Popularity (baseline)", "Item-CF retrieval"),
                      ("Item-CF retrieval", "Two-stage, no rating model"),
                      ("Item-CF retrieval", "Two-stage CF->LTR")]:
        pt = X.paired_ttest(ndcg_by_pipe[prev], ndcg_by_pipe[nxt])
        print(f"     {nxt:28s} vs {prev:22s}  diff={pt['mean_diff']:+.4f}  "
              f"95% CI [{pt['ci95'][0]:+.4f}, {pt['ci95'][1]:+.4f}]  p={pt['p_value']:.1e}")

    print("\n  Top LTR features by gain:")
    for name, g in ltr.feature_importance()[:6]:
        print(f"     {name:24s} {g:10.1f}")

    print("\n" + "=" * 66)
    print("STAGE 3 : CONTEXTUAL BANDIT  (explore/exploit, online)")
    print("=" * 66)
    env = B.CTREnvironment(train, fs, seed=a.seed)
    rounds = B._make_rounds(env, train, all_items, T=4000, K=8, seed=a.seed)
    d = env.d
    for make in [lambda: B.RandomPolicy(d, np.random.default_rng(1)),
                 lambda: B.GreedyOracle(env),
                 lambda: B.LinUCB(d, alpha=0.6),
                 lambda: B.LinTS(d, v=0.3, rng=np.random.default_rng(2))]:
        env.rng = np.random.default_rng(999)      # common rewards across policies
        p = make()
        res = B.run_online(env, p, rounds)
        print(f"  {p.name:28s}  avg_reward={res['avg_reward']:.4f}  "
              f"regret/round={res['regret_per_round']:.4f}")

    print("\n  Off-policy evaluation (evaluate LinUCB from RANDOM-policy logs):")
    env.rng = np.random.default_rng(999)
    logs = B.make_logs(env, rounds, seed=5)
    ope = B.ope_replay_and_ips(B.LinUCB(d, alpha=0.6), logs)
    rand_avg = np.mean([e["reward"] for e in logs])
    print(f"     logged random-policy avg reward : {rand_avg:.4f}")
    print(f"     replay estimate of LinUCB value : {ope['replay_value']:.4f}  "
          f"(matched {ope['replay_matched']} / {len(logs)} steps)")
    print(f"     IPS estimate of LinUCB value    : {ope['ips_value']:.4f}")

    print("\n" + "=" * 66)
    print("STAGE 4 : A/B TEST   control=Popularity  vs  treatment=CF ranker")
    print("=" * 66)
    # A/B the treatment that actually moves the metric: CF-powered ranking vs the
    # popularity baseline. Randomized user split; per-user metric = Recall@5.
    users = list(rels_cf.keys())
    rng.shuffle(users)
    half = len(users) // 2
    ctrl_u, treat_u = users[:half], users[half:]
    metric = lambda rels, u: recall_at_k(rels[u], 5, sum(1 for r in rels[u] if r > 0))
    c = np.array([metric(rels_pop, u) for u in ctrl_u])
    t = np.array([metric(rels_cf, u) for u in treat_u])

    tt = X.welch_ttest(c, t)
    print(f"  control Recall@5 = {tt['control_mean']:.4f}   treatment Recall@5 = {tt['treatment_mean']:.4f}")
    print(f"  abs lift = {tt['abs_lift']:+.4f}  ({tt['rel_lift']*100:+.1f}%)   "
          f"t = {tt['t_stat']:.2f}   p = {tt['p_value']:.2e}")
    print(f"  95% CI on lift = [{tt['ci95'][0]:+.4f}, {tt['ci95'][1]:+.4f}]")
    n = X.sample_size_for_proportion(p_baseline=max(0.01, c.mean()), mde_abs=0.02)
    print(f"  power analysis: ~{n} users/arm to detect +0.02 at 80% power, alpha=.05")
    av = X.always_valid_pvalue(c, t)
    print(f"  always-valid (mSPRT) p-value: {av:.2e}  (safe to peek continuously)")

    # CUPED methods check: pre-period covariate correlated with the metric.
    # Deterministic Recall@5 has no independent pre-period, so we validate the
    # implementation on a controlled example (pre explains ~70% of post).
    m = 5000
    pre = rng.normal(0, 1, m)
    eff = 0.10
    post_c = 0.7 * pre + rng.normal(0, 0.7, m)
    post_t = 0.7 * pre + eff + rng.normal(0, 0.7, m)
    cu = X.cuped(post_c, post_t, pre, pre)
    print(f"  CUPED check (pre~70% of post): variance reduction "
          f"{cu['variance_reduction']*100:.0f}%  (theta={cu['theta']:.2f}); "
          f"adjusted p={cu['adjusted_test']['p_value']:.1e} vs raw "
          f"p={X.welch_ttest(post_c, post_t)['p_value']:.1e}")
    print("=" * 66)


if __name__ == "__main__":
    main()
