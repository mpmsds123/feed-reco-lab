# feed-reco-lab — two-stage recommender, contextual bandit, and A/B evaluation

An end-to-end **recommendation & ranking** system on the Yelp dataset:
candidate generation → learning-to-rank → contextual-bandit re-ranking → offline
A/B evaluation. Built on top of my DSCI 553 hybrid recommender
([hybrid-recommender-system](https://github.com/mpmsds123/hybrid-recommender-system)),
which predicted Yelp *ratings*; this repo uses that rating model inside a system
that **ranks** what to show and **experiments** on it — the way a production recsys/ranking team works.

Everything runs out-of-the-box on a synthetic dataset in the **exact Yelp schema**,
and swaps to the real Yelp dump with no code changes. Every number below is
computed by `run_pipeline.py`, not hard-coded — re-run with `--seed` to see it move.

---

## Architecture

```
        user + context
             │
   ┌─────────▼──────────┐   STAGE 1 — RETRIEVAL (recall)
   │  candidate sources │   • PopularityRetriever   (baseline)
   │                    │   • ItemCFRetriever        (item-based CF on "likes")
   └─────────┬──────────┘
             │  ~100 candidates
   ┌─────────▼──────────┐   STAGE 2 — LEARNING TO RANK (order)
   │   LambdaMARTRanker │   XGBoost rank:ndcg, trained on held-out ratings;
   │ (CF rank + score,  │   predicted stars come from the DSCI 553 rating
   │  predicted stars,  │   model (XGBoost + GradientBoosting ensemble)
   │  3 Yelp features)  │
   └─────────┬──────────┘
             │  ranked list
   ┌─────────▼──────────┐   STAGE 3 — CONTEXTUAL BANDIT (explore/exploit)
   │   LinUCB / LinTS   │   simulation: rewards drawn from a click model fit
   │  + off-policy eval │   on the ratings; evaluated off-policy from logs
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐   STAGE 4 — EXPERIMENTATION
   │  A/B test harness  │   offline user split: Welch / paired t-test + CI,
   │                    │   power analysis, CUPED, always-valid (mSPRT) p-values
   └────────────────────┘
```

---

## Results — REAL Yelp Open Dataset (`data/real_yelp`)

This is the actual course dataset from DSCI 553 (`yelp_train.csv` — 455,854
ratings, `yelp_val.csv` — 142,044 ratings, `business.json` — 192,609 businesses).
**`user.json` / `checkin.json` / `photo.json` / `tip.json` were not found locally**
(likely read from the course cluster, not saved), so the 10 user-profile features
and the check-in/photo/tip features fall back to defaults — the ranker below is
running on real ratings and real business attributes only, not the full 28-feature
set. That is a real, stated limitation, not a hidden one.

**Stage 1+2 — ranking quality (held-out likes, rating ≥ 4, n=11,260 users):**

| pipeline | NDCG@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| Popularity (baseline) | 0.0128 | 0.0118 | 0.033 |
| **Item-CF retrieval** | **0.0489** | **0.0404** | **0.114** |
| Two-stage CF→LambdaMART | rerun pending | | |

The two-stage row is blank because the ranker was retrained differently on
Nov 17, 2026 (see the changelog below); the real-data number needs a rerun with the
local Yelp files. The popularity and Item-CF rows use unchanged code.

→ **Item-CF retrieval is ~3.8x popularity's NDCG@10 on real Yelp** — the absolute
numbers are small because real Yelp is huge and sparse (192K businesses, most
users rate only a handful), which is expected given how sparse the real dataset is.

**Stage 3 — contextual bandit, simulated rewards (avg reward per shown item, oracle = 0.868):**

| policy | avg reward | regret/round |
|---|---|---|
| Random (logging) | 0.660 | 0.204 |
| **LinUCB** | **0.841** | 0.024 |
| **LinTS (Thompson)** | **0.835** | 0.029 |

Off-policy evaluation from *random-policy logs* recovers LinUCB's value without
ever deploying it: **replay = 0.766, IPS = 0.820** (true online ≈ 0.841).

**Stage 4 — offline A/B test (random 50/50 user split, Popularity vs CF) + methods checks:**

- lift = **+236% Recall@5** (+0.128 absolute, t = 22.18, p = 3e-106, 95% CI [+11.7%, +13.9%])
- power analysis: N/arm to detect a +0.02 lift at 80% power
- CUPED implementation check on simulated data built so the pre-period explains half the
  variance: **50% variance reduction**, as designed (adjusted p 1.4e-11 vs raw 1.8e-06)
- always-valid mSPRT p-value reported (deliberately more conservative than the
  fixed-horizon p — it allows continuous peeking without inflating false positives)

Re-run with `python run_pipeline.py --data_dir data/real_yelp` (real files are
gitignored — not checked in — so a local copy of `yelp_train.csv`,
`yelp_val.csv`, `business.json` needs to be added to that folder first). The rating
model is trained twice (on 80% and on 100% of train), so expect a longer run than the
old ~3 minutes. A synthetic fallback (`data/synthetic`, generated automatically if
missing) exists only so the repo runs for someone without the Yelp files —
the real-data numbers above are the ones that reflect actual model behavior;
the synthetic ones reflect a much smaller, generated dataset and are not
comparable.

---

## Results — synthetic data (`data/synthetic`, seed 42)

Stage 1+2 (n = 1,998 users):

| pipeline | NDCG@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| Popularity (baseline) | 0.1802 | 0.1497 | 0.3407 |
| Item-CF retrieval | 0.2315 | 0.1792 | 0.4241 |
| Two-stage, no rating model | 0.2312 | 0.1797 | 0.4198 |
| **Two-stage CF→LambdaMART** | **0.2362** | **0.1834** | **0.4253** |

Paired t-test on per-user NDCG@10 (same users, each stage vs the one before):

| comparison | diff | 95% CI | p |
|---|---|---|---|
| Item-CF vs Popularity | +0.0513 | [+0.0475, +0.0550] | 2.4e-136 |
| Two-stage, no rating model vs Item-CF | −0.0003 | [−0.0021, +0.0015] | 0.74 |
| Two-stage CF→LambdaMART vs Item-CF | +0.0046 | [+0.0024, +0.0069] | 6.2e-05 |

Every stage beats the one before it. The ranker only beats CF when it gets the
predicted star rating from the DSCI 553 rating model; without it, it ties CF.
Tree depth (2) was picked on a validation slice of the training users, not on the test set.

---

## Changelog

**Nov 17, 2026 — ranker retrained on held-out ratings; rating model connected.**
- Found: the ranker's training negatives were each user's top CF candidates they had
  not rated. Those are where their future likes sit, so the ranker learned to push CF's
  best items down (synthetic NDCG@10 0.1386, below popularity's 0.1802).
- Fixed: train split 80/20; CF and the rating model are built on the 80%; candidates
  are labelled with the held-out 20% ratings.
- CF score now enters as rank and per-user normalised score, because raw co-like
  counts scale with data size and differ between the 80% and 100% models.
- Added the DSCI 553 rating model (`src/rating_model.py`) as a ranker feature.
- Added a paired t-test of each stage against the one before.
- Bandit stage made reproducible (user order was taken from an unordered set).

---

## Run it

```bash
pip install -r requirements.txt
python data/make_synthetic_yelp.py --data_dir data/synthetic
python run_pipeline.py --data_dir data/synthetic
```

**Swap in real Yelp** (no code changes): put the real `yelp_train.csv`,
`business.json`, `user.json`, `checkin.json`, `photo.json`, `tip.json` in a folder
and run `python run_pipeline.py --data_dir /path/to/real_yelp`. The feature builder
in `features.py` is lifted directly from the DSCI 553 `competition.py`, so the
28 features are identical.

---

## Design Rationale

- **Two stages, not one.** Retrieval optimizes recall cheaply over the whole catalog;
  ranking optimizes order over ~100 candidates with an expensive model. Separating
  them is what lets ranking run at scale without scoring the whole catalog with
  the expensive model.
- **Item-CF on likes only, not all engagement.** Co-*liking* (rating ≥ 4) reflects taste; co-*engagement*
  including dislikes is just exposure. Building CF on likes sharpens the signal.
- **Ranker labels from held-out ratings.** The ranker is trained on the same kind of
  list it reorders at serve time (CF candidates the user has not rated), labelled by
  ratings the retriever never saw. Treating unrated candidates as known negatives
  teaches the model to bury future likes.
- **Rating model as a ranker feature.** CF says what people like you engage with; the
  rating model says how many stars you would give. The ranker needs both to beat CF.
- **No raw volume features in the ranker.** Popularity, check-in, photo and tip counts
  track exposure, not taste, and hurt held-out ranking.
- **LinUCB vs ε-greedy.** UCB explores where it's *uncertain* (optimism under
  uncertainty), not uniformly at random — lower regret. LinTS does the same via
  posterior sampling.
- **Off-policy evaluation.** Not every candidate policy can be A/B tested in
  production. Replay/IPS estimate a new policy's value from logs of the current
  one, unbiased when the logging propensities are known.
- **CUPED and always-valid p-values.** CUPED removes pre-experiment variance,
  reaching significance with fewer users; mSPRT allows stopping early without the
  peeking problem that inflates false positives in fixed-horizon tests.

---

## Files

```
data/make_synthetic_yelp.py   latent-factor generator, real Yelp schema
src/features.py               28-feature builder (from DSCI 553 competition.py)
src/retrieval.py              PopularityRetriever, ItemCFRetriever
src/rating_model.py           star-rating ensemble from DSCI 553 competition.py
src/ranker.py                 LambdaMART LTR on held-out labels, depth chosen on validation
src/bandit.py                 CTR env, LinUCB, LinTS, replay + IPS off-policy eval
src/experiment.py             Welch + paired t-test, power, CUPED, mSPRT
src/eval_metrics.py           NDCG / Recall / MAP
run_pipeline.py               end-to-end demo + report
```
