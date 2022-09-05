# Import dependencies
from vizdoom import *
import numpy as np
import cv2
# Usefull constants
from utils.constants import SCENARIO_PATH, ACTIONS, TARGET_SHAPES, SKIP_FRAMES
# OpenAI Gym
from gym import Env
from gym.spaces import Discrete, Box

class GymEnv(Env):
    def __init__(self, render=False, hd=False):
        # Inherit from Env
        super().__init__()
        
        # Game Initialization
        self.game = vizdoom.DoomGame()
        self.game.load_config(SCENARIO_PATH)

        self.actions_length = len(ACTIONS)
        self.observation_space = Box(low=0, high=255, shape=TARGET_SHAPES, dtype=np.uint8)
        self.action_space = Discrete(self.actions_length)
        
        # Transform Actions list to a vector list.
        # Ex: [MOVE_RIGHT, MOVE_LEFT, ATTACK] => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.actions = np.identity(self.actions_length, dtype=np.uint8)
        
        # Render frame configuration
        if render == False:
            self.game.set_window_visible(False)
            
        # High Definition configuration
        if hd==True:
            self.game.set_screen_resolution(ScreenResolution.RES_800X600)
            self.game.set_render_hud(True)
        
        self.game.init()

    def step(self, action):
        # Check if the action is legit
        assert action >= 0 & action <= self.actions_length
        
        # Make an action, then wait {{SKIP_FRAMES}} frames
        reward = self.game.make_action(self.actions[action], SKIP_FRAMES)
        
        # Get all the stuff we need to return
        state = self.game.get_state()
        is_done = self.game.is_episode_finished()
        
        # Check state for prevent error
        if state: 
            screen = state.screen_buffer # Screenshot
            screen = self.grayscale(screen)
            ammo = state.game_variables[0] # Ammo Left
            info = ammo
        else:
            screen = np.zeros(self.observation_space.shape) # List of 0 with shape of the observation_space
            info = 0
        
        info = {"info":info}
        return screen, reward, is_done, info
    
    # Reset the game
    def reset(self):
        self.game.new_episode()
        screen = self.game.get_state().screen_buffer
        screen = self.grayscale(screen)
        return screen
    
    # Grayscale the game frame and resize it for better computation time
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (TARGET_SHAPES[1], TARGET_SHAPES[0]), interpolation=cv2.INTER_CUBIC)
        screen = np.reshape(resize, TARGET_SHAPES)
        return screen
    
    # Close the current game session
    def close(self):
        self.game.close()