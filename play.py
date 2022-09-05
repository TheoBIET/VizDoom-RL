import time

# GymEnv Class
from classes.GymEnv import GymEnv
# Usefull constants
from utils.constants import N_EVAL_EPISODES, DEFEND_CENTER_SCENARIO
# Stable Baselines
from stable_baselines3 import PPO
# Import Evaluation Policy to test our agent
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH='./models/defend_center_0509'
SCENARIO=DEFEND_CENTER_SCENARIO

model = PPO.load(MODEL_PATH)
env = GymEnv(scenario_path=SCENARIO, render=True, hd=True)

mean_reward, _ = evaluate_policy(model, 
                                 env, 
                                 n_eval_episodes=N_EVAL_EPISODES)