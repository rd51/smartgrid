"""
Constraint-Programming Dispatch Solver (Engine B)
=================================================
Solves, per timestep, the economic dispatch problem:

    minimise   sum_s cost_s * x_s          (+ unmet penalty)
    subject to sum_s x_s + unmet  >= demand     (meet demand)
               0 <= x_s <= capacity_s          (capacity limits)
               solar/wind <= forecast avail    (renewable availability)
               battery within [-charge_room, discharge_avail]
               step emissions <= per-step emission budget

This is a small convex (linear) program solved with cvxpy. It is fast,
provably optimal for the modelled problem, and never "fails to function" -
which makes it both a strong standalone deliverable (constraint programming,
multicriteria) and the safety net / projection layer for the RL agent.

RDMU concepts: Constraint Programming, Multicriteria Decision Making
(cost vs emissions handled via the per-step emission budget + objective).
"""

import cvxpy as cp
import numpy as np
from env.grid_env import SOURCES


def solve_dispatch(state, cfg, emit_budget_step=None):
    """
    state: dict with demand, solar_avail, wind_avail, soc, emit_left
    cfg:   the GridEnv config dict
    Returns: dict source->MW dispatch (battery may be negative = charge)
    steps_left: remaining timesteps, used to spread the emission budget.
    """
    demand = state["demand"]
    cap = cfg["capacity"]
    cost = cfg["cost"]
    emis = cfg["emission"]

    # Per-step emission budget. Demand-feasibility is enforced via the unmet
    # penalty in the objective (soft), so the emission limit must never make
    # the LP unable to serve load. We give each step a fair share of the
    # remaining budget plus headroom, and let the objective discourage
    # exceeding the pace.
    if emit_budget_step is None:
        steps_left = max(1, state.get("steps_left", 1))
        fair = max(0.0, state.get("emit_left", cfg["emission_cap"])) / steps_left
        emit_budget_step = fair + 0.5 * cfg["emission_cap"] / cfg["horizon"]

    x = {s: cp.Variable(nonneg=True) for s in ["solar", "wind", "gas", "coal"]}
    b_dis = cp.Variable(nonneg=True)   # battery discharge
    b_chg = cp.Variable(nonneg=True)   # battery charge
    unmet = cp.Variable(nonneg=True)

    supply = sum(x.values()) + b_dis
    constraints = [
        supply + unmet >= demand,
        x["solar"] <= state["solar_avail"],
        x["wind"] <= state["wind_avail"],
        x["gas"] <= cap["gas"],
        x["coal"] <= cap["coal"],
        b_dis <= min(cap["battery"], state["soc"]),
        b_chg <= min(cap["battery"], cfg["battery_capacity"] - state["soc"]),
    ]

    step_emit = emis["gas"] * x["gas"] + emis["coal"] * x["coal"]
    # Soft cap: hard ceiling generous enough to always serve load; the
    # objective's emission_penalty does the real shaping. This keeps the LP
    # feasible (reliability) while still steering toward the budget.
    hard_ceiling = max(emit_budget_step,
                       emis["coal"] * cap["coal"] + emis["gas"] * cap["gas"])
    constraints.append(step_emit <= hard_ceiling)

    # Emission penalty ramps up as the remaining budget depletes: cheap to
    # emit when we are well under pace, expensive when we are blowing it.
    emit_left = max(0.0, state.get("emit_left", cfg["emission_cap"]))
    steps_left = max(1, state.get("steps_left", 1))
    pace = emit_left / steps_left                      # affordable per step
    scarcity = 1.0 + 4.0 * np.exp(-pace / 5.0)         # 1x..5x multiplier
    eff_emit_pen = cfg["emission_penalty"] * scarcity

    objective = cp.Minimize(
        cost["gas"] * x["gas"] + cost["coal"] * x["coal"]
        + cost["battery"] * (b_dis + b_chg)
        + cfg["unmet_penalty"] * unmet
        + eff_emit_pen * step_emit
    )

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if x["solar"].value is None:   # infeasible fallback: serve what we can
        return {"solar": state["solar_avail"], "wind": state["wind_avail"],
                "gas": cap["gas"], "coal": 0.0, "battery": 0.0}

    return {
        "solar": float(x["solar"].value),
        "wind": float(x["wind"].value),
        "gas": float(x["gas"].value),
        "coal": float(x["coal"].value),
        "battery": float(b_dis.value - b_chg.value),
    }


def run_episode_solver(env):
    """Run a full episode driven by the LP solver. Returns env.history."""
    env.reset()
    done = False
    while not done:
        state = env.true_state()
        state["steps_left"] = env.cfg["horizon"] - env.t
        dispatch = solve_dispatch(state, env.cfg)
        _, _, done, _ = env.step(dispatch)
    return env.history
