from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt.config import settings
from supply_chain_opt.optimizer.cost_builder import CFLPData, build_problem_data
from supply_chain_opt.optimizer.cflp import SolveResult, solve_cflp


@pytest.fixture(scope="module")
def tiny_problem() -> CFLPData:
    # Facility 0 is ~111 km from both demand nodes; facilities 1 and 2 are thousands of km away.
    # With K=1, optimal solution must open facility 0.
    facility_df = pd.DataFrame({
        "lat": [0.0, 50.0, -50.0],
        "lon": [0.0, 0.0, 0.0],
        "n_points": [100, 50, 50],
    })
    demand_df = pd.DataFrame({
        "Order Region": ["RegionA", "RegionB"],
        "demand": [1000.0, 1000.0],
        "lat": [1.0, -1.0],
        "lon": [0.0, 0.0],
    })
    cost_matrix = np.array([
        [111.0, 111.0],
        [5500.0, 5600.0],
        [5600.0, 5500.0],
    ])
    return CFLPData(
        facility_df=facility_df,
        demand_df=demand_df,
        cost_matrix=cost_matrix,
        capacities=np.array([2000.0, 2000.0, 2000.0]),
        fixed_costs=np.array([1e5, 1e5, 1e5]),
        demands=np.array([1000.0, 1000.0]),
        n_facilities=3,
        n_demand=2,
    )


@pytest.fixture(scope="module")
def infeasible_problem() -> CFLPData:
    # One facility with capacity 1, demand 1e9 — provably infeasible.
    facility_df = pd.DataFrame({"lat": [0.0], "lon": [0.0], "n_points": [1]})
    demand_df = pd.DataFrame({
        "Order Region": ["RegionA"],
        "demand": [1e9],
        "lat": [1.0],
        "lon": [0.0],
    })
    return CFLPData(
        facility_df=facility_df,
        demand_df=demand_df,
        cost_matrix=np.array([[111.0]]),
        capacities=np.array([1.0]),
        fixed_costs=np.array([1e5]),
        demands=np.array([1e9]),
        n_facilities=1,
        n_demand=1,
    )


@pytest.fixture(scope="module")
def tiny_result(tiny_problem: CFLPData) -> SolveResult:
    return solve_cflp(tiny_problem, max_open_facilities=1, cost_per_km=1.0)


@pytest.fixture(scope="module")
def real_data() -> CFLPData:
    return build_problem_data(settings)


@pytest.fixture(scope="module")
def real_result(real_data: CFLPData) -> SolveResult:
    return solve_cflp(
        real_data,
        max_open_facilities=settings.cflp_max_open_facilities,
        cost_per_km=settings.cflp_cost_per_km,
        solver_backend=settings.cflp_solver_backend,
        time_limit_s=settings.cflp_solver_time_limit_s,
    )


class TestSolveCflp:
    def test_status_is_optimal(self, tiny_result: SolveResult) -> None:
        assert tiny_result.status == "OPTIMAL"

    def test_open_facilities_within_cap(self, tiny_result: SolveResult) -> None:
        assert tiny_result.n_facilities_open <= 1

    def test_demand_satisfaction(
        self, tiny_problem: CFLPData, tiny_result: SolveResult
    ) -> None:
        for j in range(tiny_problem.n_demand):
            total = tiny_result.assignments[:, j].sum()
            assert abs(total - 1.0) < 1e-6, f"Demand node {j}: assignment sum = {total}"

    def test_capacity_not_exceeded(
        self, tiny_problem: CFLPData, tiny_result: SolveResult
    ) -> None:
        for i in range(tiny_problem.n_facilities):
            served = (tiny_result.assignments[i, :] * tiny_problem.demands).sum()
            assert served <= tiny_problem.capacities[i] + 1e-6

    def test_objective_positive(self, tiny_result: SolveResult) -> None:
        assert tiny_result.objective_value > 0

    def test_assignments_in_unit_interval(self, tiny_result: SolveResult) -> None:
        assert (tiny_result.assignments >= -1e-9).all()
        assert (tiny_result.assignments <= 1.0 + 1e-9).all()


class TestSolveCflpInfeasible:
    def test_reports_infeasible(self, infeasible_problem: CFLPData) -> None:
        result = solve_cflp(infeasible_problem, max_open_facilities=1, cost_per_km=1.0)
        assert result.status in {"INFEASIBLE", "UNKNOWN", "NOT_SOLVED", "ABNORMAL"}


class TestSolveCflpRealData:
    def test_problem_dimensions(self, real_data: CFLPData) -> None:
        assert real_data.n_facilities == settings.cflp_n_facility_clusters
        assert real_data.n_demand == 23

    def test_solve_reaches_optimal(self, real_result: SolveResult) -> None:
        assert real_result.status in {"OPTIMAL", "FEASIBLE"}

    def test_open_facility_count_within_cap(self, real_result: SolveResult) -> None:
        assert real_result.n_facilities_open <= settings.cflp_max_open_facilities

    def test_demand_satisfaction_real(
        self, real_data: CFLPData, real_result: SolveResult
    ) -> None:
        for j in range(real_data.n_demand):
            total = real_result.assignments[:, j].sum()
            assert abs(total - 1.0) < 1e-4, f"Demand node {j}: assignment sum = {total}"
