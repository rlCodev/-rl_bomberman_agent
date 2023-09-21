from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from BombermanEnv import BombermanEnv

env = BombermanEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

model = A2C("MlpPolicy", env).learn(total_timesteps=1000)
