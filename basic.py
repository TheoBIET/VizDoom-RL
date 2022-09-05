# GymEnv Class
from classes.GymEnv import GymEnv
# Callback class
from classes.Callback import TrainAndLoggingCallback
# Usefull constants
from utils.constants import BASIC_SCENARIO, SAVE_MODEL_FREQUENCY, MODEL_TYPE, LEARNING_RATE, N_STEPS, N_TIMESTEPS
# Stable Baselines
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO

CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'

env = GymEnv(scenario_path=BASIC_SCENARIO)
callback = TrainAndLoggingCallback(check_freq=SAVE_MODEL_FREQUENCY, save_path=CHECKPOINT_DIR)

model = PPO(MODEL_TYPE,
            env,
            tensorboard_log=LOG_DIR,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS)

model.learn(total_timesteps=N_TIMESTEPS, callback=callback)