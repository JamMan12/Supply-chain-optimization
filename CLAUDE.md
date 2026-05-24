# Supply Chain Network Optimizer — CLAUDE.md

## Project Overview

This is a portfolio project that builds a **resilient supply chain network optimizer** combining
Machine Learning with Mixed-Integer Linear Programming (MILP). The core idea: train a binary
delay classifier to predict shipment delay probabilities on each route, then feed those
probabilities directly into an OR-Tools solver as risk-adjusted transportation costs. The
optimizer (a Capacitated Facility Location Problem) then naturally avoids high-risk lanes when
selecting which facilities to open and how to route demand.

The project is developed in three phases:

- **Phase 1 — Baseline MILP**: Solve the CFLP using raw distance-based costs. No ML yet.
- **Phase 2 — ML Model**: Train a binary delay classifier on the DataCo dataset. Output is
  `P(delay_ij)` per route.
- **Phase 3 — Risk-Adjusted Optimizer**: Replace `c_ij` with `c_ij_adjusted` and re-solve.
  Compare Phase 1 vs Phase 3 solutions to demonstrate resilience improvement.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Solver | OR-Tools (CP-SAT or MILP via `ortools.linear_solver`) |
| ML | scikit-learn / XGBoost / LightGBM (binary classifier) |
| API | FastAPI |
| Frontend | React + Next.js |
| Notebooks | Jupyter (exploration only — see Off-limits) |
| Formatting | Black (line length 88), isort |
| Type checking | mypy (strict mode on public APIs) |
| Testing | pytest |

---

## Project Structure

```
SupplyChainOpt/
├── src/
│   └── supply_chain_opt/
│       ├── data/            # DataLoader, preprocessing, feature pipeline
│       ├── models/          # Delay classifier: train, evaluate, persist
│       ├── optimizer/       # CFLP formulation, OR-Tools wrapper, cost builder
│       ├── api/             # FastAPI app, routers, request/response schemas
│       └── config.py        # Pydantic Settings — all paths and hyperparams here
├── frontend/                # Next.js app (network map dashboard)
├── notebooks/               # Exploration and EDA only
├── tests/
│   ├── unit/
│   └── integration/
├── data/
│   ├── raw/                 # Original DataCo CSV — never modified
│   └── processed/           # Cleaned and feature-engineered outputs
├── models/                  # Serialized ML model artifacts (e.g., .pkl, .json)
├── pyproject.toml           # Project metadata, dependencies, tool config
└── CLAUDE.md
```

---

## Data

Primary dataset: **DataCo Smart Supply Chain** (Kaggle).

- Store raw files in `data/raw/` — treat as immutable.
- All preprocessing writes to `data/processed/`.
- If augmenting with additional public datasets (e.g., port delay indices, weather), add a new
  subfolder under `data/raw/` and document the source in `src/supply_chain_opt/data/README.md`.
- All file paths are resolved via `config.py` (Pydantic `BaseSettings`). Never hardcode paths
  anywhere else.

---

## Mathematical Formulation — CFLP

### Sets
- `i = 1…n` — potential facility locations
- `j = 1…m` — customers (demand nodes)

### Parameters
| Symbol | Meaning |
|---|---|
| `d_j` | Demand of customer `j` |
| `f_i` | Fixed cost of opening facility `i` |
| `c_ij` | Cost of serving customer `j` from facility `i` (distance-based in Phase 1) |
| `C_i` | Capacity of facility `i` |
| `K` | Maximum number of facilities to open (configurable) |

### Decision Variables
- `x_ij ∈ [0, 1]` — fraction of customer `j`'s demand fulfilled by facility `i`
- `y_i ∈ {0, 1}` — 1 if facility `i` is opened

### Objective
```
Minimize Z = Σ_i Σ_j (c_ij · d_j · x_ij)  +  Σ_i (f_i · y_i)
```
Minimize total transportation cost (demand-weighted) plus total fixed facility opening costs.

### Constraints
1. **Demand satisfaction**: `Σ_i x_ij = 1` for all `j`
2. **Capacity**: `Σ_j (x_ij · d_j) ≤ C_i · y_i` for all `i`
3. **Facility cap**: `Σ_i y_i ≤ K`
4. **Binary**: `y_i ∈ {0, 1}`
5. **Non-negativity**: `x_ij ≥ 0`

### Phase 3 — Risk-Adjusted Cost
Replace `c_ij` with:
```
c_ij_adjusted = c_ij × (1 + P(delay_ij) × penalty_factor)
```
Where `P(delay_ij)` is the ML classifier's output for route `(i, j)`. The model structure does
not change — only the cost parameter is updated. `penalty_factor` is a configurable scalar.

---

## ML Model — Delay Classifier

- **Task**: Binary classification — predict whether a shipment on route `(i, j)` will be delayed.
- **Target variable**: Derived from DataCo's `Late_delivery_risk` or similar delay indicator.
- **Model family**: Start with XGBoost or LightGBM (tabular data; interpretable feature
  importances are a portfolio plus).
- **Output used by solver**: `predict_proba()[:, 1]` — the probability of delay per route.
- **Artifacts**: Save trained model to `models/delay_classifier.pkl` (or `.json` for XGBoost
  native format). Include feature names and version metadata alongside.
- Evaluation metrics: AUC-ROC, precision-recall curve, and calibration curve (calibration
  matters because the probability feeds directly into costs).

---

## API — Single `/solve` Endpoint

FastAPI exposes one primary endpoint:

```
POST /solve
```

**What it does internally:**
1. Accepts network configuration (facilities, customers, demands, capacities, costs, `K`,
   `penalty_factor`).
2. Loads the trained delay classifier and scores every route `(i, j)` → `P(delay_ij)`.
3. Builds risk-adjusted cost matrix.
4. Calls OR-Tools solver.
5. Returns: open facilities, flow assignments, total cost, risk scores per route, solve time.

**Supporting endpoints:**
- `GET /health` — liveness check
- `POST /predict` — score routes with ML only (useful for frontend previews)

Request/response models are defined with Pydantic v2 in `src/supply_chain_opt/api/schemas.py`.

---

## Frontend

React + Next.js dashboard. The primary view is a **network map** showing:
- Open vs closed facilities (marker style)
- Active routes with risk score encoded as color (green → red)
- Summary panel: total cost, number of open facilities, avg delay probability

The frontend calls `POST /solve` with user-configured parameters and renders the returned
solution. Keep API calls in a dedicated `lib/api.ts` module; no fetch calls scattered in
components.

---

## Coding Conventions

### Python
- **Formatter**: Black with `line-length = 88`. Run via `black src/ tests/`.
- **Imports**: isort with Black-compatible profile (`profile = "black"` in `pyproject.toml`).
- **Type hints**: Required on all public function and method signatures. Use `mypy` for checking.
  Internal helpers may omit hints when the types are trivially obvious.
- **Docstrings**: Google style. Write them on public classes and functions only. Skip internal
  helpers. One short line max for obvious functions; no multi-paragraph docstrings.
- **Comments**: Only when the WHY is non-obvious (a hidden constraint, a workaround, an
  invariant). Never explain what the code does.
- **Tests**: pytest. Tests use real or fixture-generated data — no mocks. Integration tests hit
  the real solver and a real (small) dataset.
- **Config**: All configurable values (paths, hyperparameters, solver settings, penalty factor)
  live in `src/supply_chain_opt/config.py` via Pydantic `BaseSettings`. Read from environment
  variables or a `.env` file.

### Frontend (Next.js / TypeScript)
- TypeScript strict mode.
- Tailwind CSS for styling.
- API layer isolated in `lib/api.ts`.

### pyproject.toml tool config
```toml
[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.mypy]
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Off-Limits — Things Claude Must Never Do

1. **No logic in notebooks.** Notebooks are for exploration and EDA only. All reusable code
   lives in `src/`. If something in a notebook works, extract it to `src/`.

2. **No hardcoded paths.** Every file path goes through `config.py`. Never write a literal path
   string inside a function or module.

3. **No mock data in tests.** Tests use real data fixtures or small synthetic datasets generated
   by fixture code. Do not use `unittest.mock` to fake data shapes or return values.

4. **No premature abstraction.** Don't generalize a pattern until it appears in three or more
   concrete places. A one-off helper function is better than an over-engineered base class built
   for hypothetical future cases.

5. **No features beyond what the current phase requires.** Build Phase 1 before Phase 2 before
   Phase 3. Don't stub out Phase 3 hooks while implementing Phase 1.

6. **No backwards-compatibility shims.** If something is removed, delete it. Don't leave unused
   aliases, `# removed` comments, or deprecated wrappers.
