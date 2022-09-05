# Files Path
SCENARIO_PATH = 'scenarios/basic.cfg'
CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'
MODEL_PATH = './models/doom_0509'

# Constants variables
ACTIONS = ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK']
SHAPES = (3, 240, 320)
TARGET_SHAPES = (100, 160, 1)
SKIP_FRAMES = 4
SAVE_MODEL_FREQUENCY = 10000
MODEL_TYPE = 'CnnPolicy'
LEARNING_RATE = 0.0001
N_STEPS = 2048
N_TIMESTEPS = 100000
N_EVAL_EPISODES = 20