import time

# GymEnv Class
from classes.GymEnv import GymEnv
# Usefull constants
from utils.constants import MODEL_PATH, N_EVAL_EPISODES
# Stable Baselines
from stable_baselines3 import PPO
# Import Evaluation Policy to test our agent
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load(MODEL_PATH)
env = GymEnv(render=True, hd=True)

mean_reward, _ = evaluate_policy(model, 
                                 env, 
                                 n_eval_episodes=N_EVAL_EPISODES)