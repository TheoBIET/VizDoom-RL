import numpy as np
import cv2
# VizDoom
from vizdoom import *
# OpenAI Gym
from gym import Env
from gym.spaces import Discrete, Box
from utils.constants.constants import *

class GymEnv(Env):
    def __init__(self, 
                 scenario_path, 
                 n_actions=N_ACTIONS, 
                 damage_taken_delta_w=DAMAGE_TAKEN_DELTA_W,
                 hitcount_delta_w=HITCOUNT_DELTA_W,
                 ammo_delta_w=AMMO_DELTA_W,
                 render=False, 
                 hd=False):
        # Inherit from Env
        super().__init__()
        
        # Game Initialization
        self.game = DoomGame()
        self.game.load_config(scenario_path)
        
        self.n_actions = n_actions
        self.observation_space = Box(low=LOW, high=HIGH, shape=TARGET_SHAPES, dtype=np.uint8)
        self.action_space = Discrete(n_actions)
        
        # Create a vector list represent the actions.
        self.actions = np.identity(n_actions, dtype=np.uint8)
        
        # Game Variables : HEALTH, DAMAGE_TAKEN, HITCOUNT, SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52
        
        # Reward Shaping Weight
        self.damage_taken_delta_w = damage_taken_delta_w
        self.hitcount_delta_w = hitcount_delta_w
        self.ammo_delta_w = ammo_delta_w
        
        # Render frame configuration
        if render == False:
            self.game.set_window_visible(False)
            
        # High Definition configuration
        if hd == True:
            self.game.set_screen_resolution(ScreenResolution.RES_800X600)
            self.game.set_render_hud(True)
        
        self.game.init()

    def step(self, action):
        # Check if the action is legit
        assert action >= 0 & action <= self.n_actions
        
        # Make an action, then wait {{SKIP_FRAMES}} frames
        game_reward = self.game.make_action(self.actions[action])
        reward = game_reward
        
        # Get all the stuff we need to return
        state = self.game.get_state()
        is_done = self.game.is_episode_finished()
        
        # Check state for prevent error
        if state: 
            screen = state.screen_buffer # Screenshot
            screen = self.grayscale(screen)
            
            # Reward Shaping
            game_variables = state.game_variables
            health, damage_taken, hitcount, ammo = game_variables
            
            # Calculate Rewards Delta
            # sd=10dp & d=20dp => -20 + 10 = -10
            # Make understand the agent the need to protect themselves from damage
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            
            # 0 & 1 = 1 - 0 = 1
            # Make understand the agent that they have to shoot at the targets
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            
            # 60 & 59 => 59 - 60 => -1
            # Make understand the agent that it is necessary to avoid shooting in the void
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            reward = (game_reward + 
                      damage_taken_delta*self.damage_taken_delta_w + 
                      hitcount_delta*self.hitcount_delta_w + 
                      ammo_delta*self.ammo_delta_w)
            
            info = {
                "ammo": ammo,
                "health": health,
                "damage_taken": damage_taken,
                "hitcount": hitcount
            }
        else:
            screen = np.zeros(self.observation_space.shape) # List of 0 with shape of the observation_space
            info = dict()
            
        return screen, reward, is_done, info
    
    # Reset the game
    def reset(self):
        # Reset Games Variables before start a new game
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52
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