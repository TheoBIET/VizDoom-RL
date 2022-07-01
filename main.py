from vizdoom import *

import numpy as np
import random
import time

# Constants variables
SCENARIO = 'github/ViZDoom/scenarios/basic.cfg'
ACTIONS = ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK']
EPISODES_NUMBER = 1

# Game Initialization
actions_choices = np.identity(len(ACTIONS), dtype=np.uint8)
game = vizdoom.DoomGame()
game.load_config(SCENARIO)
game.init()

# Games Loop
for i in range(EPISODES_NUMBER):
    # Create new game
    game.new_episode()
    
    while not game.is_episode_finished():
        action = random.choice(actions_choices) # Get random action (= identity matrix)
        action_index = list(action).index(1) # Index of 1 in the matrix
        state = game.get_state()
        screen = state.screen_buffer # Screen state
        infos = state.game_variables # Game variables
        # Make a random action then wait for 4 frames for logging
        reward = game.make_action(action, 4)

        print(f'Action: {ACTIONS[action_index]} \nReward: {reward}')
        time.sleep(0.02)

    total_reward = game.get_total_reward()
    print(f'Game {i+1} finished. Total Reward: {total_reward}')