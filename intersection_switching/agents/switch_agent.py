import numpy as np
import queue
import operator

from agents.agent import Agent

class SwitchAgent(Agent):
    """
    The class defining an agent which controls the traffic lights using the analytical approach
    from Helbing, Lammer's works
    """
    def __init__(self, env, ID='', **kwargs):
        """
        initialises the Analytical Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(env, ID)

        self.clearing_phase = None
        self.clearing_time = 0

        self.action_queue = queue.Queue()
        self.agents_type = 'switch'
        self.approach_lanes = []
        for phase in self.phases.values():
            for movement_id in phase.movements:
                self.approach_lanes += self.movements[movement_id].in_lanes
        self.init_phases_vectors()

    def init_phases_vectors(self):
        """
        initialises vector representation of the phases
        :param eng: the cityflow simulation engine
        """
        idx = 1
        vec = np.zeros(len(self.phases))
        # self.clearing_phase.vector = vec.tolist()
        for phase in self.phases.values():
            vec = np.zeros(len(self.phases))
            if idx != 0:
                vec[idx-1] = 1
            phase.vector = vec.tolist()
            idx += 1

    def observe(self, veh_distance):
        observations = self.phase.vector + self.get_vehicle_approach_states()
        return np.array(observations)

    def get_vehicle_approach_states(self):
        ROADLENGTH = 300 # meters, hardcoded
        VEHLENGTH = 5 # meters, hardcoded

        lane_vehicles = self.env.lane_vehs
        state_vec = []
        for lane_id in self.approach_lanes:
            speeds = []
            waiting_times = []
            for veh_id in lane_vehicles[lane_id]:
                vehicle = self.env.vehicles[veh_id]
                speeds.append(self.env.veh_speeds[veh_id])
                waiting_times.append(vehicle.stopped)
            density = len(lane_vehicles[lane_id]) * VEHLENGTH / ROADLENGTH
            ave_speed = np.mean(speeds or 0)
            ave_wait = np.mean(waiting_times or 0)
            state_vec += [density, ave_speed, ave_wait]
        return state_vec

    def aggregate_votes(self, votes, agg_func=None):
        """
        Aggregates votes using the `agg_func`.
        :param votes: list of tuples of (vote, weight). Vote is a boolean to switch phases
        :param agg_func: aggregates votes and weights and returns the winning vote.
        """
        choices = {0: 0, 1: 0}
        if agg_func is None:
            agg_func = lambda x: x
        for vote, weight in votes:
            choices[vote] += agg_func(weight)
        return max(choices, key=choices.get)


    def switch(self, eng, lane_vehs, lanes_count):
        curr_phase = self.phase.ID
        action = abs(curr_phase-1) # ID zero is clearing
        super().apply_action(eng, action, lane_vehs, lanes_count)

    def apply_action(self, eng, votes, lane_vehs, lanes_count):
        will_switch = np.random.random()>0.5
        if will_switch:
            curr_phase = self.phase.ID
            action = abs(curr_phase-1) # ID zero is clearing
        else:
            action = self.phase.ID
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        super().apply_action(eng, action, lane_vehs, lanes_count)
