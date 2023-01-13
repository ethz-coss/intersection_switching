from engine.cityflow.intersection import Movement, Phase
import numpy as np
from gym import spaces
import random

class VehicleAgent:
    """
    The base clase of an Agent, Learning and Analytical agents derive from it, basically defines methods used by both types of agents
    """

    def __init__(self, env, ID):
        """
        initialises the Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        """
        self.ID = ID
        self.env = env

        self.stopped = 0
        self.distance = 0
        self.total_rewards = []
        self.start_time = 0

        n_actions = 2 # (binary choice?)
        n_states = 10 # TODO: edit
        self.observation_space = spaces.Box(low=np.zeros(n_states), 
                                            high=np.ones(n_states),
                                            dtype=float)

        self.action_space = spaces.Discrete(n_actions)

    def get_vote(self):
        # if self.stopped:
        #     return 'wait'
        # else:
        #     return 'speed'
        return self.preference

    def set_objective(self, eng, phase):
        """
        sets the phase of the agent to the indicated phase
        """
        pass

    def get_reward(self, lanes_count):
        """
        gets the reward of the agent in the form of pressure
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        return 0 # TODO: revise

    def reset(self):
        """
        Resets the set containing the vehicle ids for each movement and the arr/dep vehicles numbers as well as the waiting times
        the set represents the vehicles waiting on incoming lanes of the movement
        """
        self.stopped = 0
        self.distance = 0
        self.total_rewards = []
        self.start_time = 0

    def observe(self, eng, time, lanes_count, lane_vehs, veh_distance):
        raise NotImplementedError

    def apply_action(self, eng, action, time, lane_vehs, lanes_count):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        pass

    def update(self):
        if (self.action_type == 'update' and
                self.env.time == (self.last_act_time+self.clearing_time)):
            self.set_phase(self.env.eng, self.chosen_phase)
            self.action_type = "act"

    def calculate_reward(self, lanes_count):
        reward = self.get_reward(lanes_count)
        self.total_rewards += [reward]
        self.reward_count += 1
        return reward
