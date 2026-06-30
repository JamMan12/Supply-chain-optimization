from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pulp

from supply_chain_opt.config import settings
from supply_chain_opt.optimizer.cost_builder import CFLPData, build_problem_data

# Keyed on prob.sol_status, which distinguishes optimal vs feasible (time-limited).
_SOL_STATUS_MAP = {
    1: "OPTIMAL",
    2: "FEASIBLE",
    0: "NOT_SOLVED",
    -1: "INFEASIBLE",
    -2: "UNBOUNDED",
}

_HAS_SOLUTION = {1, 2}


@dataclass
class SolveResult:
    status: str
    objective_value: float
    open_facilities: list[int]
    assignments: np.ndarray  # shape (n_facilities, n_demand), x_ij values
    solve_time_s: float
    n_facilities_open: int


def _get_solver(backend: str, time_limit_s: int) -> pulp.LpSolver:
    if backend.upper() == "HIGHS":
        return pulp.HiGHS(msg=0, timeLimit=time_limit_s)
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)


def solve_cflp(
    data: CFLPData,
    max_open_facilities: int,
    cost_per_km: float,
    solver_backend: str = "HIGHS",
    time_limit_s: int = 60,
) -> SolveResult:
    """Solve the CFLP as a MILP using PuLP.

    Minimizes total transport cost + fixed facility costs subject to demand
    satisfaction, capacity, and a cap on the number of open facilities.
    """
    n_fac = data.n_facilities
    n_dem = data.n_demand

    prob = pulp.LpProblem("cflp", pulp.LpMinimize)

    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_fac)]
    x = [
        [pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1) for j in range(n_dem)]
        for i in range(n_fac)
    ]

    # Objective: transport cost + fixed opening cost
    prob += (
        pulp.lpSum(data.fixed_costs[i] * y[i] for i in range(n_fac))
        + pulp.lpSum(
            cost_per_km * data.cost_matrix[i, j] * data.demands[j] * x[i][j]
            for i in range(n_fac)
            for j in range(n_dem)
        )
    )

    # Demand satisfaction: sum_i x_ij == 1 for each j
    for j in range(n_dem):
        prob += pulp.lpSum(x[i][j] for i in range(n_fac)) == 1

    # Capacity: sum_j x_ij * d_j <= C_i * y_i for each i
    for i in range(n_fac):
        prob += (
            pulp.lpSum(x[i][j] * data.demands[j] for j in range(n_dem))
            <= data.capacities[i] * y[i]
        )

    # Facility cap: sum_i y_i <= K
    prob += pulp.lpSum(y[i] for i in range(n_fac)) <= max_open_facilities

    solver = _get_solver(solver_backend, time_limit_s)
    t0 = time.perf_counter()
    prob.solve(solver)
    solve_time_s = time.perf_counter() - t0

    status_str = _SOL_STATUS_MAP.get(prob.sol_status, "UNKNOWN")
    has_solution = prob.sol_status in _HAS_SOLUTION

    if has_solution:
        open_facilities = [i for i in range(n_fac) if pulp.value(y[i]) > 0.5]
        assignments = np.array(
            [[pulp.value(x[i][j]) for j in range(n_dem)] for i in range(n_fac)]
        )
        obj_val = pulp.value(prob.objective)
    else:
        open_facilities = []
        assignments = np.zeros((n_fac, n_dem))
        obj_val = float("nan")

    return SolveResult(
        status=status_str,
        objective_value=obj_val,
        open_facilities=open_facilities,
        assignments=assignments,
        solve_time_s=solve_time_s,
        n_facilities_open=len(open_facilities),
    )


def run(save: bool = True) -> SolveResult:
    """Load processed CFLP data, solve the baseline model, and optionally save results."""
    data = build_problem_data(settings)
    result = solve_cflp(
        data,
        max_open_facilities=settings.cflp_max_open_facilities,
        cost_per_km=settings.cflp_cost_per_km,
        solver_backend=settings.cflp_solver_backend,
        time_limit_s=settings.cflp_solver_time_limit_s,
    )

    print(f"Status         : {result.status}")
    print(f"Objective      : {result.objective_value:,.0f}")
    print(f"Facilities open: {result.n_facilities_open} of {settings.cflp_max_open_facilities}")
    print(f"Solve time     : {result.solve_time_s:.3f}s")
    print()
    for idx in result.open_facilities:
        row = data.facility_df.iloc[idx]
        print(f"  Facility {idx:>2}: lat={row['lat']:.4f}  lon={row['lon']:.4f}  "
              f"(cluster of {int(row['n_points'])} raw points)")

    if save:
        payload = {
            "status": result.status,
            "objective_value": result.objective_value,
            "open_facilities": result.open_facilities,
            "n_facilities_open": result.n_facilities_open,
            "solve_time_s": result.solve_time_s,
            "assignments": result.assignments.tolist(),
        }
        settings.cflp_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.cflp_results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {settings.cflp_results_path}")

    return result


if __name__ == "__main__":
    run(save=True)
