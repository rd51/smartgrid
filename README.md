# ⚡ Smart Energy Grid Optimizer — RDMU Topic 5

Decision-making software module for **Reasoning and Decision Making under
Uncertainty (MAIB DSC 103)**. Balances renewable (solar, wind) and
conventional (gas, coal) generation plus battery storage to meet a
**stochastic** electricity demand, under emission and capacity constraints.

## RDMU concepts demonstrated (4+ required)
1. **Markov Decision Process** — dispatch framed as state→action→reward.
2. **Sequential Decision Making** — battery storage couples decisions across hours.
3. **Methods for Estimation** — the agent acts on *noisy forecasts* of demand &
   renewables, not ground truth (the "uncertainty").
4. **Multicriteria Decision Making** — cost vs. emissions vs. reliability,
   visualised as a Pareto frontier.
5. *(bonus)* **Utility Theory** — reward = utility over the weighted objective.

## Two decision engines
| Engine | Method | Role |
|---|---|---|
| **B — Constraint solver** | Linear program (cvxpy) | Provably optimal dispatch; the reliable baseline |
| **A — SAC agent** | Soft Actor-Critic + **constraint-projection** safety layer | Learned policy kept feasible by projection ("safe RL") |

## Run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py          # interactive dashboard
python -m agents.train_sac 8000         # (re)train the RL agent
```

## Structure
```
env/grid_env.py        stochastic grid simulator (MDP)
agents/solver.py       constraint-programming dispatch (Engine B)
agents/rl_agent.py     Gym wrapper + projection layer
agents/train_sac.py    SAC training -> assets/sac_grid.zip
dashboard/app.py       Streamlit control-room dashboard
notebooks/colab.ipynb  Colab-runnable reproduction
```
