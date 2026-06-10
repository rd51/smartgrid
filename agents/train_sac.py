"""
Train SAC on the grid env and save a checkpoint.
Kept short by design (demo within time budget). The constraint-projection
layer makes even a lightly-trained agent produce feasible, sensible dispatch.
"""
import sys, time
from stable_baselines3 import SAC
from agents.rl_agent import GridGymEnv


def train(timesteps=8000, save_path="assets/sac_grid"):
    env = GridGymEnv(project=True)
    model = SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                buffer_size=20000, batch_size=256, train_freq=1,
                gradient_steps=1, gamma=0.98)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(save_path)
    print(f"trained {timesteps} steps in {time.time()-t0:.1f}s -> {save_path}.zip")
    return model


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    train(n)
