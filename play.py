import time

# GymEnv Class
from classes.GymEnv import GymEnv
# Usefull constants
from utils.constants import N_EVAL_EPISODES
# Stable Baselines
from stable_baselines3 import PPO
# Import Evaluation Policy to test our agent
from stable_baselines3.common.evaluation import evaluate_policy

SCENARIO_PATH = 'scenarios/defend_the_center.cfg'
MODEL_PATH = 'models/defend_center_0509'

################ DEFEND THE CENTER ################
# SCENARIO_PATH = 'scenarios/defend_the_center.cfg'
# MODEL_PATH = 'models/defend_center_0509'
##################### BASIC ######################
# SCENARIO_PATH = 'scenarios/basic.cfg'
# MODEL_PATH = 'models/basic_0509'
##################################################

model = PPO.load(MODEL_PATH)
env = GymEnv(scenario_path=SCENARIO_PATH, render=True, hd=True)

mean_reward, _ = evaluate_policy(model, 
                                 env, 
                                 n_eval_episodes=N_EVAL_EPISODES)