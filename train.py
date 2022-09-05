# GymEnv Class
from classes.GymEnv import GymEnv
# Callback class
from classes.Callback import TrainAndLoggingCallback
# Usefull constants
from utils.constants import SAVE_MODEL_FREQUENCY, MODEL_TYPE, LEARNING_RATE, N_STEPS, N_TIMESTEPS, CLIP_RANGE, GAMMA
# Stable Baselines
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO

# Select same level with multiples degrees of difficulty for Curriculumn Learning
SCENARIO_PATH_START='scenarios/deadly_corridor_s'
DIFFICULTY_COUNT=5

for i in range(DIFFICULTY_COUNT):
    path_index = i + 1
    checkpoint_dir = f'./train/deadly_corridor_s{path_index}'
    log_dir = f'./logs/deadly_corridor_s{path_index}'
    
    # Create new env with new configuration file
    env = GymEnv(
        scenario_path=f'{SCENARIO_PATH_START}{path_index}.cfg',
        n_actions=7,
        render=True
    )
    
    # Check environnement
    env_checker.check_env(env)

    callback = TrainAndLoggingCallback(check_freq=SAVE_MODEL_FREQUENCY, save_path=checkpoint_dir)

    if path_index == 1:
        model = PPO(MODEL_TYPE,
                    env,
                    tensorboard_log=log_dir,
                    verbose=1,
                    learning_rate=LEARNING_RATE,
                    n_steps=N_STEPS,
                    clip_range=CLIP_RANGE,
                    gamma=GAMMA,
                    gae_lambda=GAE_LAMBDA)

        model.learn(total_timesteps=N_TIMESTEPS, callback=callback)
    else:
        model.set_env(env)
        model.learn(total_timesteps=N_TIMESTEPS, callback=callback)