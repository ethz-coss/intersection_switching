from engine.cityflow.intersection import Phase
from agents.switch_agent import SwitchAgent
import numpy as np

class FixedAgent(SwitchAgent):

    def __init__(self, env, ID='', **kwargs):
        super().__init__(env, ID, **kwargs)
        self.agents_type = 'fixed'
        self.phase = self.phases[np.random.randint(1, 8)]

    def choose_act(self, eng, time):
        loop_start_action_id = self.phase.ID
        phaseID = loop_start_action_id

        green_time = 0

        while green_time == 0:
            phaseID = max((phaseID+1) % len(self.phases), 1) # skip clearing phase
            chosen_phase = self.phases[phaseID]
            green_time = 30
            # green_time = int(np.max([self.movements[move_id].get_green_time(time, [], eng) for move_id in chosen_phase.movements]))
            # if chosen_phase.ID == loop_start_action_id:
            #     green_time = 10
            #     break
            # print(f'time: {self.env.time} ID: {self.ID} phase: {phaseID}, green: {green_time}')
        return phaseID, green_time
      

    # def observe(self, veh_distance):
    #     return None