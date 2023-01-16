import cityflow

import numpy as np
import random
import os
import functools
from utils import flow_creator
from collections import Counter

from engine.cityflow.intersection import Lane
from gym import utils
import gym
from pettingzoo.utils.env import ParallelEnv, AECEnv
from pettingzoo.utils import agent_selector
from agents.vehicle_agent import VehicleAgent
from agents.switch_agent import SwitchAgent 

class Environment(gym.Env):
    """
    The class Environment represents the environment in which the agents operate in this case it is a city
    consisting of roads, lanes and intersections which are controled by the agents
    """

    metadata = {"name": "cityflow"}

    def __init__(self, args=None, n_vehs=[30,15], reward_type='speed', ID=0):
        """
        initialises the environment with the arguments parsed from the user input
        :param args: the arguments input by the user
        :param n_actions: the number of possible actions for the learning agent, corresponds to the number of available phases
        :param n_states: the size of the state space for the learning agent
        """
        print('configfile', args.sim_config,os.path.abspath(args.sim_config))
        flow_creator(os.path.dirname(os.path.abspath(args.sim_config)), n_vehs=n_vehs)
        self.n_vehs = n_vehs
        self.eng = cityflow.Engine(args.sim_config, thread_num=os.cpu_count())
        self.ID = ID
        self.num_sim_steps = args.num_sim_steps
        self.update_freq = args.update_freq      # how often to update the network
        self.batch_size = args.batch_size

        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.eps_update = args.eps_update

        self.eps = self.eps_start
        self.reward_type = reward_type

        self._warmup()

        self.time = 0
        random.seed(2)

        self.veh_speeds = self.eng.get_vehicle_speed()
        self.lane_vehs = self.eng.get_lane_vehicles()
        self.lanes_count = self.eng.get_lane_vehicle_count()

        self.agents_type = args.agents_type

        self.action_freq = 5  # typical update freq for agents

        self.intersection_ids = ['intersection_0_0'] # single intersection only
        self.intersections = {}
        for intersection_id in self.intersection_ids:
            self.intersections[intersection_id] = SwitchAgent(self, ID=intersection_id,
                                                        in_roads=self.eng.get_intersection_in_roads(intersection_id),
                                                        out_roads=self.eng.get_intersection_out_roads(intersection_id),
                                                        lr=args.lr, batch_size=args.batch_size)


        self.agents = list(self.intersections.values())
        self.agent_ids = list(self.intersections.keys())
        self._agents_dict = self.intersections

        n_states = self.agents[0].observation_space.shape[0]

        self.observations = {agent_id: np.zeros(n_states) for agent_id in self.agent_ids}
        self.actions = {agent_id: None for agent_id in self.agent_ids}
        self.action_probs = {
            agent_id: None for agent_id in self.agent_ids}
        self.rewards = {agent_id: None for agent_id in self.agent_ids}
        self._cumulative_rewards = {agent_id: None for agent_id in self.agent_ids}
        self.dones = {a_id: False for a_id in self.agent_ids}
        self.dones['__all__'] = False
        self.infos =  {agent: False for agent in self.agent_ids}

        self.mfd_data = []
        self.agent_history = []

        self.lanes = {}

        for lane_id in self.eng.get_lane_vehicles().keys():
            self.lanes[lane_id] = Lane(self.eng, ID=lane_id)

        # metrics
        self.speeds = []
        self.stops = []
        self.stops_idx = 0 # start measuring reward from this index
        self.speeds_idx = 0 # start measuring reward from this index
        self.waiting_times = []

    def _warmup(self):
        for _ in range(200):
            self.eng.next_step()
            if len(self.eng.get_vehicles())>=sum(self.n_vehs):
                break

        veh_dict = self.eng.get_vehicles()
        if len(veh_dict)<sum(self.n_vehs):
            print(f'WARNING: {len(veh_dict)}/{sum(self.n_vehs)} vehicles generated. Increase warmup period.')
            
        self.vehicles = {}
        for veh_id in veh_dict:
            self.vehicles[veh_id] = VehicleAgent(self, veh_id)

    @property
    def observation_space(self):
        return self.agents[0].observation_space
    
    @property
    def action_space(self):
        return self.agents[0].action_space

    def observation_spaces(self, ts_id):
        return self._agents_dict[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self._agents_dict[ts_id].action_space

    def step(self, actions):
        assert actions is not None
        self._apply_actions(actions)
        self.sub_steps()

        rewards = self._compute_rewards()
        observations = self._get_obs()
        info = self.infos
        dones = self._compute_dones()

        return observations, rewards, dones, info

    def sub_steps(self):
        time_to_act = False
        self.stops_idx = 0
        self.speeds_idx = 0
        while not time_to_act:
            self.eng.next_step()
            self.time += 1

            stops = 0
            self.veh_speeds = self.eng.get_vehicle_speed()
            self.lane_vehs = self.eng.get_lane_vehicles()
            self.lanes_count = self.eng.get_lane_vehicle_count()

            # required to track distance of periodic trips
            for veh_id, speed in self.veh_speeds.items():
                self.vehicles[veh_id].distance += speed
                self.vehicles[veh_id].speeds.append(speed)

            for lane_id, lane in self.lanes.items():
                lane.update_flow_data(self.eng, self.lane_vehs)
                lane.update_speeds(self, self.lane_vehs[lane_id], self.veh_speeds)

            for veh_id, speed in self.veh_speeds.items():
                veh = self.vehicles[veh_id]
                if speed <= 0.1:
                    veh.wait += 1
                    if veh.wait == 1:
                        stops += 1  # first stop
                        veh.stops += 1
                elif speed > 0.1 and veh.wait:
                    self.waiting_times.append(veh.wait)
                    veh.wait_times.append(veh.wait)
                    veh.wait = 0
            self.speeds.append(np.mean(list(self.veh_speeds.values())))
            self.stops.append(stops)
            self.stops_idx += 1
            self.speeds_idx += 1

            if self.time % self.update_freq == 0:  # TODO: move outside to training
                self.eps = max(self.eps-self.eps_decay, self.eps_end)

            for intersection in self.intersections.values():
                intersection.update()
                if intersection.time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        for intersection in self.intersections.values():
            # lane_vehicles = self.eng.get_lane_vehicles()
            # votes = []
            # for lane_id in intersection.approach_lanes:
            #     for veh_id in lane_vehicles[lane_id]:
            #         votes.append(self.vehicles[veh_id].get_vote())
            intersection.apply_action(self.eng, actions[intersection.ID],
                                   self.lane_vehs, self.lanes_count)

    def _get_obs(self):
        vehs_distance = self.eng.get_vehicle_distance()

        self.observations = {tl.ID: tl.observe(vehs_distance) for tl in self.intersections.values()}
        return self.observations

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.intersection_ids}
        dones['__all__'] = self.time > self.num_sim_steps
        return dones

    def _compute_rewards(self):
        self.rewards = {tl.ID: tl.calculate_reward(self.lanes_count, type=self.reward_type) for tl in self.intersections.values()}
        return self.rewards

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self.observations[agent]

    def reset(self, seed=None, options=None):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        # super().reset(seed=seed)
        if seed is None:
            seed = random.randint(1, 1e6)
        self.eng.reset(seed=False)
        self.eng.set_random_seed(seed)

        self._warmup()
        self.eng.set_save_replay(True)

        self.time = 0
        for agent in self.agents:
            agent.reset()

        for lane_id, lane in self.lanes.items():
            lane.speeds = []
            lane.dep_vehs_num = []
            lane.arr_vehs_num = []
            lane.prev_vehs = set()

        self.veh_speeds = self.eng.get_vehicle_speed()
        self.lane_vehs = self.eng.get_lane_vehicles()
        self.lanes_count = self.eng.get_lane_vehicle_count()
        self.waiting_times = []
        self.speeds = []
        self.stops = []


        obs = self._get_obs()
        info = {}

        return obs

    def get_mfd_data(self, time_window=60):
        mfd_detailed = {}

        for lane_id in self.eng.get_lane_vehicles().keys():
            mfd_detailed[lane_id] = {"speed": [], "density": []}

        for lane_id, lane in self.lanes.items():
            data = mfd_detailed[lane_id]
            speed = data['speed']
            density = data['density']

            _lanedensity = np.subtract(
                lane.arr_vehs_num, lane.dep_vehs_num).cumsum()
            for t in range(self.num_sim_steps):
                time_window = min(time_window, t+1)
                idx_start = t
                idx_end = t+time_window

                s = np.mean(sum(lane.speeds[idx_start:idx_end], []))
                d = _lanedensity[idx_start:idx_end].mean() / lane.length

                speed.append(s)
                density.append(d)

        return mfd_detailed

    def vote_drivers(self):
        votes = {'speed': 0, 'wait': 0, 'stops': 0}
        # votes = []
        for intersection in self.intersections.values():
            lane_vehicles = self.eng.get_lane_vehicles()
            for lane_id in intersection.approach_lanes:
                for veh_id in lane_vehicles[lane_id]:
                    # votes[self.vehicles[veh_id].get_vote()] += 1
                    votes[self.vehicles[veh_id].preference] += 1

        # return Counter(votes)
        return votes

    def assign_driver_preferences(self, preferences_dict):
        for veh_id, preference in preferences_dict.items():
            self.vehicles[veh_id].preference = preference
