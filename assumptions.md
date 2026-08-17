# Project Assumptions

Modeling decisions that are not self-evident from the code or data.
Each entry states the value chosen and why.

---

## CFLP — Phase 1

| Assumption | Value | Reasoning |
|---|---|---|
| Facility candidate set | KMeans clusters of raw lat/lon points (k=20) | 11,835 unique warehouse coordinates produce too many binary variables for a direct MILP; clustering to 20 representative centers keeps the problem trivially small while preserving geographic diversity |
| Max open facilities (K) | 5 | Arbitrary starting point — in a real deployment this comes from budget or operational constraints |
| Facility capacity | 0.3 × total demand per facility | Ensures feasibility at K=5 (5 × 0.3 = 1.5× total demand) while preventing any single facility from satisfying all demand alone, which would make the capacity constraint irrelevant |
| Fixed cost per facility | 1e9 (same units as demand × km) | Scaled to be in the same order of magnitude as transport costs so the open/close tradeoff is visible to the solver; not a real facility cost |
| Transport cost metric | Haversine distance (km) × cost_per_km | Haversine is a reasonable great-circle proxy for shipping distance at global scale; cost_per_km=1.0 is a neutral scalar — absolute value does not affect optimizer decisions, only the ratio to fixed cost matters |
| Demand node coordinates | Hand-specified approximate geographic centroids per Order Region | DataCo contains no delivery lat/lon; geocoding Order City adds an external API dependency; approximate centroids are sufficient because centroid errors of ~200 km are negligible relative to the thousands of km separating global regions |

---

## ML Delay Classifier — Phase 2

| Assumption | Value | Reasoning |
|---|---|---|
| Train/test split | 80/20, stratified on target | 180,519 rows is large enough that a 20% (~36k row) held-out set gives stable AUC/PR/calibration estimates; stratification guards the mild 54.8/45.2 imbalance without needing resampling |
| Random state | 42, shared across split, both estimators, and CV | Single seed keeps train/eval reproducible end-to-end; no need for per-component seeds at this scale |
| Resampling for class imbalance | None (no SMOTE/undersampling) | 54.8/45.2 split is mild; resampling would add a train/inference asymmetry that complicates calibration — directly counter to the requirement that probabilities stay well-calibrated for Phase 3 |
| Categorical encoding | `OneHotEncoder(handle_unknown="ignore")`, sparse | `Order Country`'s 164 categories are affordable as sparse one-hot at 180k rows; target/ordinal encoding would need fold-aware fitting to avoid leakage — unjustified complexity for Phase 2 |
| Numeric feature scaling | None (passthrough) | XGBoost and LightGBM are tree-based and scale-invariant; a scaler would add pipeline complexity with no accuracy benefit |
| Hyperparameter tuning | `GridSearchCV`, 5-fold, exhaustive over 2-3 params per family (≤18 combos), scored on ROC-AUC | Grid small enough that exhaustive search is affordable and reproducible; ROC-AUC is threshold-independent and directly relevant to how probabilities are ranked/used downstream |
| Model selection between XGBoost/LightGBM | Both trained and persisted; a comparison report is written, no single model is promoted to a canonical `delay_classifier.pkl` name | Promotion to "the" model consumed by the solver is a Phase 3 concern — Phase 2 stays self-contained and avoids building Phase 3 hooks prematurely |
| Extra evaluation metric | Brier score, alongside AUC-ROC/PR/calibration | Standard single-number companion to a calibration curve; cheap to compute, directly answers whether predicted probabilities are trustworthy before they're used as Phase 3 cost multipliers |
| Hyperparameter grid (round 2) | Widened complexity ceiling (`max_depth` 5-9, `num_leaves` 31-127, `n_estimators` 200-400) + added regularization knobs (`min_child_weight`, `min_child_samples`) | Round 1 grid search picked the most complex option in every case, an ambiguous signal of whether that was a true optimum; round 2 confirms genuine improvement (test AUC +0.004 to +0.005, Brier improved for both) with no overfitting signal — test score exceeded CV score for both models. Both models again saturated at the new ceiling and picked the least-restrictive regularization value, so the true optimum may still be higher; not pursued further given diminishing returns |
| `route_historical_delay_rate` feature | Point-in-time, Bayesian-smoothed historical delay rate per `(Market, Order Region)` lane, `smoothing_k=20` | Two different model families converged on nearly identical AUC despite a widened tuning grid — a sign the bottleneck was feature information, not model capacity. No existing feature captured "how has this lane historically performed," typically the strongest predictor in delay-prediction problems. See `data/README.md` for the leakage-safety construction (per-row prior-only aggregation, smoothing anchor also point-in-time). Deliberately grouped by `Market`/`Order Region` rather than the Phase 1 k-means facility clusters, so it doesn't couple Phase 2 to Phase 1's arbitrary `k` — flagged consequence: `P(delay_ij)` will vary by destination region but not by facility identity until a facility-level feature is added. **Outcome**: test AUC +0.019 (xgboost, 0.7609→0.7800) and +0.025 (lightgbm, 0.7645→0.7894), Brier improved for both, calibration unchanged — a larger gain than either round of hyperparameter tuning combined, confirming the ceiling was feature information rather than model capacity |

---

## Risk-Adjusted Optimizer — Phase 3

<!-- to be filled in -->
