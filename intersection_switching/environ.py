import cityflow

import numpy as np
import random
import os
import functools
from utils import flow_creator, config_creator
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

    def __init__(self, args=None, reward_type='speed', ID=0):
        """
        initialises the environment with the arguments parsed from the user input
        :param args: the arguments input by the user
        :param n_actions: the number of possible actions for the learning agent, corresponds to the number of available phases
        :param n_states: the size of the state space for the learning agent
        """
        self.n_vehs = args.n_vehs

        if args.n_vehs is not None:
            print('we are here', args.n_vehs)
            sim_config = config_creator(os.path.dirname(os.path.abspath(args.sim_config)), n_vehs=args.n_vehs, reward=reward_type)
            flow_creator(os.path.dirname(os.path.abspath(args.sim_config)), n_vehs=args.n_vehs, reward=reward_type)
            self.fixed_num_vehicles = True
        else:
            self.fixed_num_vehicles = False
            sim_config = args.sim_config


        self.eng = cityflow.Engine(sim_config, thread_num=os.cpu_count())
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
            
        self.vehicles = {}
        if self.fixed_num_vehicles:
            self._warmup()

        self.time = 0
        self.rng = np.random.default_rng(seed=args.seed)

        self.veh_speeds = self.eng.get_vehicle_speed()
        self.lane_vehs = self.eng.get_lane_vehicles()
        self.lanes_count = self.eng.get_lane_vehicle_count()

        self.agents_type = args.agents_type

        self.action_freq = 10  # typical update freq for agents

        self.intersection_ids = [x for x in self.eng.get_intersection_ids()
                                if not self.eng.is_intersection_virtual(x)]
        # self.intersection_ids = ['intersection_0_0'] # single intersection only
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
        self.infos = {agent: False for agent in self.agent_ids}
        self.total_points = args.total_points
        self.scenario = args.scenario

        self.mfd_data = []
        self.agent_history = []

        self.lanes = {}

        for lane_id in self.lane_vehs.keys():
            self.lanes[lane_id] = Lane(self.eng, ID=lane_id)

        # metrics
        self.speeds = []
        self.stops = []
        self.stops_idx = 0 # start measuring reward from this index
        self.speeds_idx = 0 # start measuring reward from this index
        self.waiting_times = []

    def _warmup(self):
        for _ in range(1000):
            self.eng.next_step()
            if self.fixed_num_vehicles:
                if len(self.eng.get_vehicles())>=sum(self.n_vehs):
                    break
        veh_dict = self.eng.get_vehicles()
        if self.fixed_num_vehicles:
            if len(veh_dict)<sum(self.n_vehs):
                print(f'WARNING: {len(veh_dict)}/{sum(self.n_vehs)} vehicles generated. Increase warmup period.')
        
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
                if veh_id not in self.vehicles:
                    self.vehicles[veh_id] = VehicleAgent(self, veh_id) # TODO: remove old vehicles
                    if self.weights is not None:
                        self.assign_driver_preferences([veh_id], self.pref_types, weights=self.weights)
                    else:
                        self.assign_driver_preferences([veh_id], self.pref_types, total_points=self.total_points, scenario=self.scenario)
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
        for tl_id, action in actions.items():
            intersection = self._agents_dict[tl_id]
            if intersection.time_to_act:
                intersection.apply_action(self.eng, action,
                                    self.lane_vehs, self.lanes_count)

    def _get_obs(self):
        vehs_distance = self.eng.get_vehicle_distance()

        self.observations.update({tl.ID: tl.observe(
            vehs_distance) for tl in self.intersections.values() if tl.time_to_act})
        return self.observations.copy()

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.intersection_ids}
        dones['__all__'] = self.time > self.num_sim_steps
        return dones

    def _compute_rewards(self):
        self.rewards.update({tl.ID: tl.calculate_reward(self.lanes_count, type=self.reward_type) for tl in self.intersections.values() if tl.time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.intersections[ts].time_to_act}

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

        self.vehicles = {}
        if self.fixed_num_vehicles:
            self._warmup()
        # self.eng.set_save_replay(True)

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

        for lane_id in self.lane_vehs.keys():
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

    def distribute_points(self, vehicle_ids, pref_types, total_points, scenario):
        preferences_dict = {}

        for i, veh_id in enumerate(vehicle_ids):
            prob = self.rng.random()
            if scenario == 'bipolar':  # Bipolar Preference Distribution
                if prob < 0.5:
                    stop_points = self.rng.normal(0.8, 0.05) * total_points
                    points = [0, stop_points, total_points - stop_points]
                else:
                    points = self.rng.multinomial(total_points, [0, 0.5, 0.5])

            elif scenario == 'balanced_mild':  # Balanced Mild Polarization
                if prob < 0.5:
                    stop_points = self.rng.normal(0.6, 0.05) * total_points
                    points = [0, stop_points, total_points - stop_points]
                else:
                    wait_points = self.rng.normal(0.6, 0.05) * total_points
                    points = [0, total_points - int(wait_points), int(wait_points)]

            elif scenario == 'majority_mild':  # Majority-Minority Mild Polarization
                if prob < 0.6:
                    stop_points = self.rng.normal(0.6, 0.05) * total_points
                    points = [0, stop_points, total_points - stop_points]
                else:
                    wait_points = self.rng.normal(0.6, 0.05) * total_points
                    points = [0, total_points - int(wait_points), int(wait_points)]

            elif scenario == 'majority_extreme':  # Extreme Majority-Minority Polarization
                if prob < 0.2:
                    stop_points = self.rng.normal(0.95, 0.025) * total_points
                    points = [0, stop_points, total_points - stop_points]
                else:
                    wait_points = self.rng.normal(0.6, 0.05) * total_points
                    points = [0, total_points - int(wait_points), int(wait_points)]

            elif scenario == 'debug_cumulative_majority':  # debug case
                if prob < 0.5:
                    points = [0, 1, 0]
                else:
                    points = [0, 0, 1]

            preferences_dict[veh_id] = {pref: point for pref, point in zip(pref_types, points)}
            self.vehicles[veh_id].preference = preferences_dict[veh_id]  # Assign the preferences to each vehicle.

        return preferences_dict

    def vote_drivers(self, total_points, point_voting=False, binary=False):
        votes = {tl_id: {'speed': 0, 'wait': 0, 'stops': 0} for tl_id, tl in self.intersections.items()}

        lane_vehicles = self.lane_vehs
        for tl_id, intersection in self.intersections.items():
            for lane_id in intersection.approach_lanes:
                for veh_id in lane_vehicles[lane_id]:
                    max_points = max(self.vehicles[veh_id].preference.values())
                    # Sum up the scores based on the preferences of the vehicles.
                    for pref_type, _points in self.vehicles[veh_id].preference.items():
                        points = _points
                        if binary: # binarizes to total_points
                            points = total_points*(_points==max_points)
                        votes[tl_id][pref_type] += points
        return votes

    def assign_driver_preferences(self, vehicle_ids, pref_types, weights=None, total_points=None, scenario=None):
        if total_points and scenario:  # If point-based preference is enabled
            preferences_dict = self.distribute_points(vehicle_ids, pref_types, total_points, scenario)
        else:
            print("WARNING SHOULD NOT BE HAPPENING")
            choice = self.rng.choice(pref_types, p=weights)
            preferences_dict = {id: {i: int(i == choice) for i in pref_types} for id in vehicle_ids}

            for veh_id, preference in preferences_dict.items():
                self.vehicles[veh_id].preference = preference

    def get_driver_satisfactions(self, agent_id, raw_net):
        satisfactions = []
        lane_vehicles = self.lane_vehs
        for lane_id in self.intersections[agent_id].approach_lanes:
            for veh_id in lane_vehicles[lane_id]:
                score = 0
                for pref_type, points in self.vehicles[veh_id].preference.items():
                    if raw_net["reference"] == raw_net[pref_type]:
                        score += points
                satisfaction = score/sum(self.vehicles[veh_id].preference.values())
                satisfactions.append(satisfaction)
                self.vehicles[veh_id].satisfactions.append(satisfaction)
        return satisfactions