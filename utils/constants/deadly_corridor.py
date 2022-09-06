SCENARIO_PATH = 'utils/scenarios/deadly_corridor_s2.cfg' # Game Scenario Path
LOG_DIR = 'logs/deadly_corridor' # Tensorboard Log Directory
CHECKPOINT_DIR = 'train/deadly_corridor' # Checkpoint Directory
MODEL_PATH = 'utils/models/doom_model_390000' # Model Path
MODEL_NAME = 'reward' # Model Name
N_ACTIONS = 7
IS_REWARD_SHAPED = True # Is the reward shaped?
IS_CURRICULUM = True # Is the curriculum learning enabled?
CURRICULUM_PATHS = [
    'utils/scenarios/deadly_corridor_s1.cfg',
    'utils/scenarios/deadly_corridor_s2.cfg',
    'utils/scenarios/deadly_corridor_s3.cfg',
    'utils/scenarios/deadly_corridor_s4.cfg',
    'utils/scenarios/deadly_corridor_s5.cfg',
]

# PPO Model Hyperparameters
LEARNING_RATE = 0.00001
CLIP_RANGE = .1
GAMMA = .9
GAE_LAMBDA = .99
N_STEPS = 8192
N_TIMESTEPS = 500000

# Reward Shaping weights
HITCOUNT_DELTA_W=200
DAMAGE_TAKEN_DELTA_W=10
AMMO_DELTA_W=5