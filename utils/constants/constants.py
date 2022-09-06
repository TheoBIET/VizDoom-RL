# Constants variables for the GymEnv Class will never change
MODEL_TYPE = 'CnnPolicy'
N_ACTIONS = 3 # # Number of actions in the scenario
TARGET_SHAPES = (100, 160, 1) # Target shape for the CNN
SAVE_MODEL_FREQUENCY = 10000 # Frequency to save the model
N_EVAL_EPISODES = 20 # Number of episodes to evaluate the model
VERSBOSE = 2 # Verbose level for the model

# Gym Box Configuration
LOW = 0
HIGH = 255

# Default PPO Model Hyperparameters
LEARNING_RATE = 0.0003
CLIP_RANGE = .2
GAMMA = .99
GAE_LAMBDA = .95
N_STEPS = 2048
N_TIMESTEPS = 100000

# Reward Shaping weights
HITCOUNT_DELTA_W=200
DAMAGE_TAKEN_DELTA_W=10
AMMO_DELTA_W=5

# String constants for Train Class
SCENARIO_PATH = 'SCENARIO_PATH'
IS_REWARD_SHAPED = 'IS_REWARD_SHAPED'

GAME_LEVELS = [
    'basic',
    'defend_the_center',
    'deadly_corridor'
]