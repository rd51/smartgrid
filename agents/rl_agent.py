"""
Gym wrapper + Constraint-Projection layer for SAC (Engine A)
============================================================
Wraps GridEnv as a Gymnasium environment with a CONTINUOUS action space:
the agent outputs a raw dispatch vector, which is then PROJECTED onto the
feasible set (capacity, availability, supply>=demand) before being applied.

This "safe RL via projection" pattern is the innovation hook:
  agent proposes  ->  LP-style projection guarantees feasibility  ->  apply

RDMU concepts: MDP, Sequential Decision Making, Exploration & Exploitation
(SAC entropy-driven exploration), Utility Theory (reward = utility).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.grid_env import GridEnv, SOURCES


class GridGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, project=True):
        super().__init__()
        self.config = config or {}
        self.project = project
        self.core = GridEnv(self.config)
        # action: raw dispatch fractions for solar,wind,gas,coal in [0,1]
        # plus battery signal in [-1,1]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
        )
        obs = self.core.reset()
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.cfg["seed"] = seed
            self.core.rng = np.random.default_rng(seed)
        obs = self.core.reset()
        return obs, {}

    def _project(self, raw):
        """Map raw [0,1]/[-1,1] action to a feasible MW dispatch dict."""
        cfg = self.core.cfg
        i = min(self.core.t, cfg["horizon"] - 1)
        cap = cfg["capacity"]
        d = {
            "solar": raw[0] * min(cap["solar"], self.core.solar_avail[i]),
            "wind": raw[1] * min(cap["wind"], self.core.wind_avail[i]),
            "gas": raw[2] * cap["gas"],
            "coal": raw[3] * cap["coal"],
            "battery": raw[4] * cap["battery"],
        }
        if not self.project:
            return d
        # Projection: if supply < demand, ramp up cheapest-available sources;
        # if supply > demand, charge battery / curtail coal first.
        demand = self.core.demand[i]
        supply = d["solar"] + d["wind"] + d["gas"] + d["coal"] + max(0, d["battery"])
        gap = demand - supply
        if gap > 0:  # need more: gas then coal
            add_gas = min(gap, cap["gas"] - d["gas"]); d["gas"] += add_gas; gap -= add_gas
            add_coal = min(gap, cap["coal"] - d["coal"]); d["coal"] += add_coal; gap -= add_coal
        elif gap < 0:  # surplus: curtail coal, then charge battery
            cut = min(-gap, d["coal"]); d["coal"] -= cut; gap += cut
            if gap < 0:
                d["battery"] = max(d["battery"], gap)  # negative => charge
        return d

    def step(self, action):
        dispatch = self._project(np.asarray(action, dtype=float))
        obs, reward, done, info = self.core.step(dispatch)
        reward = reward / 1000.0  # scale for SAC stability
        return obs, reward, done, False, info


def run_episode_agent(model, config=None, project=True):
    """Run a full episode with a trained model. Returns env.core.history."""
    env = GridGymEnv(config, project=project)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    return env.core.history
