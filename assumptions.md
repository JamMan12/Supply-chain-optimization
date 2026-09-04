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

<!-- to be filled in -->

---

## Risk-Adjusted Optimizer — Phase 3

<!-- to be filled in -->
