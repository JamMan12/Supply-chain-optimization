from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from ortools.linear_solver import pywraplp

from supply_chain_opt.config import settings
from supply_chain_opt.optimizer.cost_builder import CFLPData, build_problem_data

_STATUS_MAP = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


@dataclass
class SolveResult:
    status: str
    objective_value: float
    open_facilities: list[int]
    assignments: np.ndarray  # shape (n_facilities, n_demand), x_ij values
    solve_time_s: float
    n_facilities_open: int


def _build_variables(
    solver: pywraplp.Solver,
    n_facilities: int,
    n_demand: int,
) -> tuple[list[pywraplp.Variable], list[list[pywraplp.Variable]]]:
    y = [solver.BoolVar(f"y_{i}") for i in range(n_facilities)]
    x = [
        [solver.NumVar(0.0, 1.0, f"x_{i}_{j}") for j in range(n_demand)]
        for i in range(n_facilities)
    ]
    return y, x


def _extract_solution(
    y: list[pywraplp.Variable],
    x: list[list[pywraplp.Variable]],
    n_facilities: int,
    n_demand: int,
    status_int: int,
    objective_value: float,
    solve_time_s: float,
) -> SolveResult:
    status = _STATUS_MAP.get(status_int, "UNKNOWN")

    if status_int in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        open_facilities = [i for i in range(n_facilities) if y[i].solution_value() > 0.5]
        assignments = np.array(
            [[x[i][j].solution_value() for j in range(n_demand)] for i in range(n_facilities)]
        )
    else:
        open_facilities = []
        assignments = np.zeros((n_facilities, n_demand))

    return SolveResult(
        status=status,
        objective_value=objective_value,
        open_facilities=open_facilities,
        assignments=assignments,
        solve_time_s=solve_time_s,
        n_facilities_open=len(open_facilities),
    )


def solve_cflp(
    data: CFLPData,
    max_open_facilities: int,
    cost_per_km: float,
    solver_backend: str = "CBC",
    time_limit_s: int = 60,
) -> SolveResult:
    """Solve the CFLP as a MILP using OR-Tools.

    Minimizes total transport cost + fixed facility costs subject to demand
    satisfaction, capacity, and a cap on the number of open facilities.
    """
    solver = pywraplp.Solver.CreateSolver(solver_backend)
    solver.SetTimeLimit(time_limit_s * 1000)

    n_fac = data.n_facilities
    n_dem = data.n_demand
    y, x = _build_variables(solver, n_fac, n_dem)

    # Objective: transport cost + fixed opening cost
    objective = solver.Objective()
    for i in range(n_fac):
        objective.SetCoefficient(y[i], data.fixed_costs[i])
        for j in range(n_dem):
            coeff = cost_per_km * data.cost_matrix[i, j] * data.demands[j]
            objective.SetCoefficient(x[i][j], coeff)
    objective.SetMinimization()

    # Demand satisfaction: sum_i x_ij = 1 for each j
    for j in range(n_dem):
        ct = solver.Constraint(1.0, 1.0)
        for i in range(n_fac):
            ct.SetCoefficient(x[i][j], 1.0)

    # Capacity (linearized): sum_j x_ij * d_j - C_i * y_i <= 0 for each i
    for i in range(n_fac):
        ct = solver.Constraint(-solver.infinity(), 0.0)
        for j in range(n_dem):
            ct.SetCoefficient(x[i][j], data.demands[j])
        ct.SetCoefficient(y[i], -data.capacities[i])

    # Facility cap: sum_i y_i <= K
    cap_ct = solver.Constraint(-solver.infinity(), float(max_open_facilities))
    for i in range(n_fac):
        cap_ct.SetCoefficient(y[i], 1.0)

    t0 = time.perf_counter()
    status_int = solver.Solve()
    solve_time_s = time.perf_counter() - t0

    obj_val = solver.Objective().Value() if status_int in (
        pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE
    ) else float("nan")

    return _extract_solution(y, x, n_fac, n_dem, status_int, obj_val, solve_time_s)


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
