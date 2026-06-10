"""
Smart Energy Grid Environment
==============================
A simulated smart-city power grid that must balance renewable (solar, wind)
and conventional (gas, coal) sources plus battery storage to meet a stochastic
demand, under emission and capacity constraints.

RDMU concepts embodied here:
  * Markov Decision Process   -> state = (demand_fc, solar_fc, wind_fc, soc, t, emit_left)
  * Sequential Decision Making -> battery carries energy across timesteps
  * Methods for Estimation     -> the agent acts on *noisy forecasts*, not truth
  * Multicriteria objective    -> reward trades off cost vs emissions vs reliability

The class is dependency-light (numpy only) so it runs in Colab, VS Code, or a
Streamlit process without gymnasium installed. A thin Gym wrapper is provided
separately for training with Stable-Baselines3.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Source catalogue. Costs in $/MWh, emissions in tCO2/MWh (illustrative).
# ----------------------------------------------------------------------------
SOURCES = ["solar", "wind", "gas", "coal", "battery"]

DEFAULT_CONFIG = {
    "horizon": 24,            # hours in an episode (one day)
    "capacity": {            # max MW dispatchable per step
        "solar": 110.0,
        "wind": 90.0,
        "gas": 90.0,
        "coal": 80.0,
        "battery": 50.0,     # max discharge (and charge) rate
    },
    "cost": {                # $/MWh marginal cost
        "solar": 0.0,
        "wind": 0.0,
        "gas": 60.0,
        "coal": 35.0,
        "battery": 5.0,      # cycling cost
    },
    "emission": {            # tCO2/MWh
        "solar": 0.0,
        "wind": 0.0,
        "gas": 0.40,
        "coal": 0.95,
        "battery": 0.0,
    },
    "battery_capacity": 120.0,   # MWh storage
    "soc_init": 60.0,            # initial state of charge MWh
    "demand_base": 120.0,        # mean demand MW
    "demand_amp": 40.0,          # daily swing amplitude
    "renewable_penetration": 1.0,  # scales solar+wind availability
    "demand_variability": 0.10,    # std of demand noise (fraction)
    "emission_cap": 250.0,         # tCO2 budget per episode
    "unmet_penalty": 500.0,        # $/MWh penalty for unserved demand
    "emission_penalty": 80.0,      # $/tCO2 penalty when over cap pace
    "forecast_noise": 0.08,        # std of forecast error (Methods for Estimation)
    "seed": 0,
}


class GridEnv:
    """Plain (numpy) environment. step() takes a dispatch dict or vector."""

    def __init__(self, config=None):
        self.cfg = dict(DEFAULT_CONFIG)
        if config:
            # shallow-merge nested dicts
            for k, v in config.items():
                if isinstance(v, dict) and k in self.cfg:
                    self.cfg[k] = {**self.cfg[k], **v}
                else:
                    self.cfg[k] = v
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.reset()

    # ------------------------------------------------------------------ #
    def _profiles(self):
        """Generate ground-truth demand & renewable availability for a day."""
        H = self.cfg["horizon"]
        t = np.arange(H)
        # demand: double-peak daily curve (morning + evening)
        base = self.cfg["demand_base"]
        amp = self.cfg["demand_amp"]
        demand = base + amp * (
            0.6 * np.sin((t - 6) / 24 * 2 * np.pi)
            + 0.4 * np.sin((t - 18) / 12 * 2 * np.pi)
        )
        noise = self.rng.normal(0, self.cfg["demand_variability"] * base, H)
        demand = np.clip(demand + noise, 10, None)

        pen = self.cfg["renewable_penetration"]
        # solar: bell around midday
        solar = self.cfg["capacity"]["solar"] * pen * np.clip(
            np.sin((t - 6) / 12 * np.pi), 0, None
        )
        solar *= self.rng.uniform(0.7, 1.0, H)
        # wind: noisier, can blow any time
        wind = self.cfg["capacity"]["wind"] * pen * np.clip(
            0.5 + 0.5 * np.sin((t - 3) / 8 * np.pi) + self.rng.normal(0, 0.2, H),
            0, 1,
        )
        return demand, solar, wind

    def reset(self):
        self.t = 0
        self.demand, self.solar_avail, self.wind_avail = self._profiles()
        self.soc = self.cfg["soc_init"]
        self.emit_used = 0.0
        self.history = []
        return self._obs()

    # ------------------------------------------------------------------ #
    def _forecast(self, arr, idx):
        """Noisy forecast of a value (Methods for Estimation)."""
        true = arr[idx]
        return max(0.0, true * (1 + self.rng.normal(0, self.cfg["forecast_noise"])))

    def _obs(self):
        """Observation = noisy forecasts + internal state (the MDP state)."""
        i = min(self.t, self.cfg["horizon"] - 1)
        return np.array([
            self._forecast(self.demand, i),
            self._forecast(self.solar_avail, i),
            self._forecast(self.wind_avail, i),
            self.soc,
            self.t / self.cfg["horizon"],
            max(0.0, self.cfg["emission_cap"] - self.emit_used),
        ], dtype=np.float32)

    def true_state(self):
        i = min(self.t, self.cfg["horizon"] - 1)
        return {
            "demand": self.demand[i],
            "solar_avail": self.solar_avail[i],
            "wind_avail": self.wind_avail[i],
            "soc": self.soc,
            "emit_left": self.cfg["emission_cap"] - self.emit_used,
        }

    # ------------------------------------------------------------------ #
    def step(self, dispatch):
        """dispatch: dict source->MW (battery may be negative = charging)."""
        cfg = self.cfg
        i = min(self.t, cfg["horizon"] - 1)
        demand = self.demand[i]

        d = {s: float(dispatch.get(s, 0.0)) for s in SOURCES}

        # clip renewables to availability, others to capacity
        d["solar"] = np.clip(d["solar"], 0, self.solar_avail[i])
        d["wind"] = np.clip(d["wind"], 0, self.wind_avail[i])
        d["gas"] = np.clip(d["gas"], 0, cfg["capacity"]["gas"])
        d["coal"] = np.clip(d["coal"], 0, cfg["capacity"]["coal"])
        # battery: + discharge (limited by soc & rate), - charge (limited by headroom & rate)
        max_dis = min(cfg["capacity"]["battery"], self.soc)
        max_chg = min(cfg["capacity"]["battery"], cfg["battery_capacity"] - self.soc)
        d["battery"] = float(np.clip(d["battery"], -max_chg, max_dis))

        supply = d["solar"] + d["wind"] + d["gas"] + d["coal"] + max(0, d["battery"])
        charge = max(0, -d["battery"])

        # update battery
        self.soc += charge - max(0, d["battery"])
        self.soc = float(np.clip(self.soc, 0, cfg["battery_capacity"]))

        unmet = max(0.0, demand - supply)

        cost = sum(cfg["cost"][s] * abs(d[s]) for s in SOURCES)
        emit = sum(cfg["emission"][s] * max(0, d[s]) for s in SOURCES)
        self.emit_used += emit

        emit_over = max(0.0, self.emit_used - cfg["emission_cap"])
        # reward: negative weighted cost (Utility Theory / multicriteria)
        reward = -(
            cost
            + cfg["unmet_penalty"] * unmet
            + cfg["emission_penalty"] * emit
            + 2.0 * cfg["emission_penalty"] * emit_over
        )

        rec = {
            "t": self.t, "demand": demand, "supply": supply,
            "unmet": unmet, "cost": cost, "emit": emit,
            "soc": self.soc, **{f"d_{s}": d[s] for s in SOURCES},
        }
        self.history.append(rec)

        self.t += 1
        done = self.t >= cfg["horizon"]
        return self._obs(), reward, done, rec
