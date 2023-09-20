import numpy as np
import queue
import operator
from gym import spaces

from agents.agent import Agent

MAXSPEED = 40/3.6 # NOTE: maxspeed is hardcoded
WAIT_THRESHOLD = 120
VEHLENGTH = 5 # meters, hardcoded

class SwitchAgent(Agent):
    """
    The class defining an agent which controls the traffic lights using the switching approach
    """
    def __init__(self, env, ID='', in_roads=[], out_roads=[], **kwargs):
        """
        initialises the Analytical Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(env, ID)

        self.stops = []
        self.unique_stops = []
        self.waits = []
        self.speeds = []

        # Needed for intersection switch only
        # self.clearing_phase = None
        # self.clearing_time = 0

        self.in_roads = in_roads
        self.out_roads = out_roads

        self.agents_type = 'switch'
        self.approach_lanes = []
        for phase in self.phases.values():
            for movement_id in phase.movements:
                self.approach_lanes += self.movements[movement_id].in_lanes
        self.init_phases_vectors()

        self.n_actions = len(self.phases)
        # nstates = 10
        nstates = len(self.get_vehicle_approach_states({}))
        # nstates = len(self.get_in_lanes_veh_num({})) + len(self.get_out_lanes_veh_num())
        self.observation_space = spaces.Box(low=np.zeros(self.n_actions+nstates), 
                                            high=np.array([1]*self.n_actions+[100]*nstates),
                                            dtype=float)

        self.action_space = spaces.Discrete(self.n_actions)


        self.action_queue = queue.Queue()
        self.last_phase_time = {phase_id: 0 for phase_id, phase in self.phases.items() if phase_id!=self.clearing_phase.ID}

    def init_phases_vectors(self):
        """
        initialises vector representation of the phases
        :param eng: the cityflow simulation engine
        """
        idx = 1
        vec = np.zeros(len(self.phases))
        self.clearing_phase.vector = vec.tolist()
        for phase in self.phases.values():
            vec = np.zeros(len(self.phases))
            if idx != 0:
                vec[idx-1] = 1
            phase.vector = vec.tolist()
            idx += 1

    def observe(self, vehs_distance):
        observations = self.phase.vector + self.get_vehicle_approach_states(vehs_distance)
        return np.array(observations)

    def get_vehicle_approach_states(self, vehs_distance):
        lane_vehicles = self.env.lane_vehs
        state_vec = []
        for lane_id in self.approach_lanes:
            # length = self.in_lanes_length[lane_id]
            speeds = []
            waits = []
            for veh_id in lane_vehicles[lane_id]:
                vehicle = self.env.vehicles[veh_id]
                speeds.append(self.env.veh_speeds[veh_id])
                waits.append(vehicle.wait)
            # density = len(lane_vehicles[lane_id]) * VEHLENGTH / length
            ave_speed = np.mean(speeds or 0)
            ave_wait = np.mean(waits or 0)
            # state_vec += [density]
            # state_vec += [ave_speed, ave_wait]
            
        in_density = self.get_in_lanes_veh_num(vehs_distance)
        out_density = self.get_out_lanes_veh_num()
        return state_vec + in_density + out_density

    def get_out_lanes_veh_num(self):
        """
        gets the number of vehicles on the outgoing lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        lanes_count = self.env.lanes_count
        lanes_veh_num = []
        for road in self.out_roads:
            lanes = self.env.eng.get_road_lanes(road)
            for lane in lanes:
                length = self.out_lanes_length[lane]
                lanes_veh_num.append(lanes_count[lane] * VEHLENGTH / length)
        return lanes_veh_num
    
    def get_in_lanes_veh_num(self, vehs_distance):
        """
        gets the number of vehicles on the incoming lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_veh: a dictionary with lane ids as keys and list of vehicle ids as values
        :param vehs_distance: dictionary with vehicle ids as keys and their distance on their current lane as value
        """
        
        lane_vehs = self.env.lane_vehs
        lanes_count = self.env.lanes_count
        lanes_veh_num = []
        for road in self.in_roads:
            lanes = self.env.eng.get_road_lanes(road)
            for lane in lanes:
                length = self.in_lanes_length[lane]
                seg1 = 0
                seg2 = 0
                seg3 = 0
                vehs = lane_vehs[lane]
                for veh in vehs:
                    if veh in vehs_distance.keys():
                        if vehs_distance[veh] / length >= (2/3):
                            seg1 += 1
                        elif vehs_distance[veh] / length >= (1/3):
                            seg2 += 1
                        else:
                            seg3 += 1

                lanes_veh_num.append((seg1 * VEHLENGTH) / (length/3))
                lanes_veh_num.append((seg2 * VEHLENGTH) / (length/3))
                lanes_veh_num.append((seg3 * VEHLENGTH) / (length/3))
        return lanes_veh_num

    def apply_action(self, eng, phase_id, lane_vehs, lanes_count):
        action = phase_id
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        super().apply_action(eng, action, lane_vehs, lanes_count)

    def measure(self):
        lane_vehicles = self.env.lane_vehs
        veh_speeds = self.env.veh_speeds
        unique_stops = 0
        stops = 0 # stopped vehicles
        waiting_times = self.waits
        for lane_id in self.approach_lanes:
            length = self.in_lanes_length[lane_id]
            for veh_id in lane_vehicles[lane_id]:
                speed = veh_speeds[veh_id]
                self.speeds.append(speed)
                veh = self.env.vehicles[veh_id]
                if speed <= 0.1:
                    veh.wait += 1
                    if veh.wait == 1:
                        unique_stops += 1  # first stop
                        veh.stops += 1
                    if veh.wait >= 1:
                        stops += 1
                elif speed > 0.1 and veh.wait:
                    waiting_times.append(veh.wait)
                    veh.wait_times.append(veh.wait)
                    veh.wait = 0
        self.unique_stops.append(unique_stops)
        self.stops.append(stops)
        return unique_stops, waiting_times

    def reset_measures(self):
        self.stops = []
        self.unique_stops = []
        self.waits = []
        self.speeds = []

    def get_reward(self, lanes_count, type='speed'):
        if type=='speed':
            if self.speeds:
                return np.mean(self.speeds)
            else:
                return 0
        if 'stops' in type:
            if 'unique' in type:
                stops = self.unique_stops
            elif 'global' in type:
                stops = self.env.stops[-self.env.stops_idx:]
            else:
                stops = self.stops
            if stops:
                return -np.mean(stops)
            else:
                return 0
            # return -np.mean(self.env.stops[-self.env.stops_idx:])
        if type=='delay':
            delays = []
            for veh_id, veh_data in self.env.vehicles.items():
                tt = self.env.time - veh_data.start_time
                dist = veh_data.distance
                delay = (tt - dist/MAXSPEED)/dist if dist!= 0 else 0
                delay *= 600 # convert to secs/600m
                delays.append(delay)
            return -np.mean(delays)
        if 'wait' in type:
            if 'local' in type:
                waiting_times = self.waits
                lane_vehicles = self.env.lane_vehs
                for lane_id in self.approach_lanes:
                    for veh_id in lane_vehicles[lane_id]:
                        veh = self.env.vehicles[veh_id]
                        # if veh.wait >= 1:
                        waiting_times.append(veh.wait)
            else:
                waiting_times = []
                for veh_id in self.env.veh_speeds.keys():
                    veh = self.env.vehicles[veh_id]
                    if veh.wait >= 1:
                        waiting_times.append(veh.wait)

            if waiting_times:
                return -np.mean(waiting_times)
            else:
                return 0
        if type=='pressure':
            return -np.abs(np.sum([x.get_pressure(lanes_count) for x in self.movements.values()]))

            
    def calculate_reward(self, lanes_count, type='speed'):

        if type == 'both':
            stops = self.get_reward(lanes_count, type='stops') / (5 * len(self.env.vehicles))
            wait = self.get_reward(lanes_count, type='wait') / 1800

            # reward = stops + wait

            reward = ((-stops)**(0.5) * ((-wait)**(0.5)))
            reward = -1000*reward

            self.total_rewards += [reward]
            self.reward_count += 1
            
            return reward
        else:
            reward = self.get_reward(lanes_count, type=type)
            self.total_rewards += [reward]
            self.reward_count += 1
            return reward

    def rescale_preferences(self, pref, qvals):
        params = {'speed': {'mean': 5,
                            'min': 0,
                            'max': 11},
                  'wait': {'mean': -0.74,
                            'min': -4.9,
                            'max': -0.56},
                  'stops': {'mean': -13.2,
                            'min': -55.6,
                            'max': -4.4}}

        alpha = 1
        # shift = qvals - qvals.max()
        shift = (qvals - params[pref]['max']) / (params[pref]['max'] - params[pref]['min'])

        return np.exp(alpha * shift)/ np.sum(np.exp(alpha*shift))
        # if pref=='speed':
        #     return qvals/(MAXSPEED * 5)
        # elif pref=='wait':
        #     return (np.clip(qvals, -WAIT_THRESHOLD * 5, 0) / (WAIT_THRESHOLD * 5)) + 1
        # elif pref=='stops':
        #     return np.clip(qvals, -len(self.env.vehicles) * 5, 0) / (len(self.env.vehicles) * 5) + 1
