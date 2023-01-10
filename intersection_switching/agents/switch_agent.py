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

        self.action_queue = queue.Queue()
        self.agents_type = 'switch'
        self.approach_lanes = []
        for phase in self.phases.values():
            for movement_id in phase.movements:
                self.approach_lanes += self.movements[movement_id].in_lanes

    def observe(self, veh_distance):
        return None

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
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        super().apply_action(eng, action, lane_vehs, lanes_count)

    def apply_action(self, eng, votes, lane_vehs, lanes_count):
        will_switch = self.aggregate_votes(votes)
        if will_switch:
            self.switch(eng, lane_vehs, lanes_count)
        else:
            super().apply_action(eng, self.phase.ID, lane_vehs, lanes_count)
