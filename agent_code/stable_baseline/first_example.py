import gymnasium as gym
import torch

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")
device = torch.device("mps")

model = A2C("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
