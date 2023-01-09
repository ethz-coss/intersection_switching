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
        self.agents_type = 'analytical'
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
        action = abs(curr_phase-2)+1 # ID zero is clearing
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        super().apply_action(eng, action, lane_vehs, lanes_count)

    def apply_action(self, eng, votes, lane_vehs, lanes_count):
        will_switch = self.aggregate_votes(votes)
        if will_switch:
            self.switch(eng, lane_vehs, lanes_count)
        else:
            super().apply_action(eng, self.phase.ID, lane_vehs, lanes_count)

        
    def choose_act(self, eng, time):
        """
        selects the next action - phase for the agent to select along with the time it should stay on for
        :param eng: the cityflow simulation engine
        :param time: the time in the simulation, at this moment only integer values are supported
        :returns: the phase and the green time
        """

        self.update_clear_green_time(time, eng)
        
        self.stabilise(time)
        if not self.action_queue.empty():
            phase, green_time = self.action_queue.get()
            return phase.ID, int(np.ceil(green_time))

        if all([x.green_time == 0 for x in self.movements.values()]):
                return self.phase.ID, 5
        
        self.update_priority_idx(time)
        phases_priority = {}
        
        for phase in self.phases.values():
            movements = [x for x in phase.movements if x not in self.clearing_phase.movements]

            phase_prioirty = 0
            for moveID in movements:
                phase_prioirty += self.movements[moveID].priority

            phases_priority.update({phase.ID : phase_prioirty})
        
        action = max(phases_priority.items(), key=operator.itemgetter(1))[0]
        chosen_phase = self.phases[action]
        if not chosen_phase.movements:
            green_time = self.clearing_time
        else:
            green_time = max(5, int(min([self.movements[x].green_time for x in chosen_phase.movements])))
        return action, green_time

    def stabilise(self, time):
        """
        Implements the stabilisation mechanism of the algorithm, updates the action queue with phases that need to be prioritiesd
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        def add_phase_to_queue(priority_list):
            """
            helper function called recursievely to add phases which need stabilising to the queue
            """
            phases_score = {}
            phases_time = {}
            for elem in priority_list:
                for phaseID in elem[0].phases:
                    if phaseID in phases_score.keys():
                        phases_score.update({phaseID : phases_score[phaseID] + 1})
                    else:
                        phases_score.update({phaseID : 1})

                    if phaseID in phases_time.keys():
                        phases_time.update({phaseID : max(phases_time[phaseID], elem[1])})
                    else:
                        phases_time.update({phaseID : elem[1]})

            if [x for x in phases_score.keys() if phases_score[x] != 0]:
                idx = max(phases_score.items(), key=operator.itemgetter(1))[0]
                self.action_queue.put((self.phases[idx], phases_time[idx]))
                return [x for x in priority_list if idx not in x[0].phases]
            else:
                return []

        T = 360
        T_max = 480
        sum_Q = sum([x.arr_rate for x in self.movements.values()])
        
        priority_list = []

        for movement in [x for x in self.movements.values() if x.ID not in self.phase.movements]:              
            Q = movement.arr_rate
            if movement.last_on_time == -1:
                waiting_time = 0
            else:
                waiting_time = time - movement.last_on_time
                
            z = movement.green_time + movement.clearing_time + waiting_time
            n_crit = Q * T * ((T_max - z) / (T_max - T))

            waiting = movement.green_time * movement.max_saturation
           
            if waiting > n_crit:
                T_res = T * (1 - sum_Q / movement.max_saturation) - self.clearing_time * len(self.movements)
                green_max = (Q / movement.max_saturation) * T + (1 / len(self.movements)) * T_res
                priority_list.append((movement, green_max))
            
        priority_list = add_phase_to_queue(priority_list)
