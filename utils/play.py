# GymEnv Class
from classes.GymEnv import GymEnv
from stable_baselines3 import PPO
# Import Evaluation Policy to test our agent
from stable_baselines3.common.evaluation import evaluate_policy

from utils.constants.constants import *

class Play():
    def __init__(self, level_name):
        self.level_name = level_name
        self.load_constants()
        
    def play(self):
        model = PPO.load(self.model_path)
        env = module.GymEnv(self.scenario_path,
                     n_actions=self.n_actions,
                     render=True,
                     hd=True
                    )
        mean_reward, _ = evaluate_policy(model, 
                                        env, 
                                        n_eval_episodes=N_EVAL_EPISODES)
        
    def load_constants(self):
        module = importlib.import_module(f"levels.{self.level_name}")
        self.model_name = module.MODEL_NAME
        self.scenario_path = module.SCENARIO_PATH
        self.model_path = module.MODEL_PATH
