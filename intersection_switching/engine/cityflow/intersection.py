import cityflow
import numpy as np

class Movement:
    """
    The class defining a Movement on an intersection, a Movement of vehicles from incoming road -> outgoing road
    """
    def __init__(self, ID, in_road, out_road, in_lanes, out_lanes, in_length, out_length, clearing_time=5, phases=[]):
        """
        initialises the Movement, the movement has a type 1, 2 or 3
        1 -> turn right, 2 -> turn left, 3 -> go straight
        :param ID: the unique (for a given intersection) ID associated with the movement
        :param in_road: the incoming road of the movement
        :param out_road: the outgoing road of the movement
        :param in_lanes: the incoming lanes of the movement
        :param out_lanes: the outgoing lanes of the movement
        :param lane_length: the length of the incoming lane (if there is more than one incoming lane we assume they have the same length)
        :param phases: the indices of phases for which the give movement is enabled
        """
        self.ID = ID
        
        self.in_road = in_road
        self.out_road = out_road

        self.in_lanes = in_lanes
        self.out_lanes = out_lanes

        self.in_length = in_length
        self.out_length = out_length
        self.phases = phases
        self.clearing_time = clearing_time

        self.arr_vehs_num = []
        self.dep_vehs_num = []

        self.move_type = None 
        self.max_saturation = 2.2
        self.max_speed = 11
        self.pass_time = int(np.ceil(self.in_length / self.max_speed))


        self.prev_vehs = set()
        self.waiting_time = 0
        self.max_waiting_time = 0
        self.waiting_time_list = []

        self.arr_rate = 0
        self.green_time = 1
        self.priority = 0
        self.last_on_time = 0

    def update_wait_time(self, time, action, phase, lanes_count):
        """
        Updates movement's waiting time - the time a given movement has waited to be enabled
        :parama time: the current time
        :param action: the phase to be chosen for the intersection in this time step
        :param phase: the phase at the intersection up till this time step
        """
        count = sum([lanes_count[x] for x in self.in_lanes])

        if self.ID not in action.movements and self.ID in phase.movements:
            self.last_on_time = time           
        elif self.ID in action.movements and self.ID not in phase.movements:
            if self.last_on_time == -1:
                self.waiting_time = 0
            else:
                self.waiting_time = time + 2 - self.last_on_time
                self.waiting_time_list.append(self.waiting_time)
                if  self.waiting_time > self.max_waiting_time:
                    self.max_waiting_time = self.waiting_time
                self.waiting_time = 0
                
            self.last_on_time = -1
        elif count == 0 and self.ID not in action.movements and self.ID not in phase.movements:
            self.last_on_time = time           

    def get_dep_veh_num(self, start_time, end_time):
        """
        gets the number of vehicles departed from the movement's lanes in a given interval
        :param start_time: the start of the time interval
        :param end_time: the end of the time interval
        :returns: the number of vehicles departed in the interval
        """
        return np.sum(self.dep_vehs_num[start_time: end_time])

    def get_arr_veh_num(self, start_time, end_time):
        """
        gets the number of vehicles arrived to the movement's lanes in a given interval
        :param start_time: the start of the time interval
        :param end_time: the end of the time interval
        :returns: the number of vehicles arrived in the interval
        """
        return np.sum(self.arr_vehs_num[start_time: end_time])

    def update_arr_dep_veh_num(self, lanes_vehs):
        """
        Updates the list containing the number vehicles that arrived and departed
        :param lanes_vehs: a dictionary with lane ids as keys and number of vehicles as values
        """
        current_vehs = set()

        for lane in self.in_lanes:
            current_vehs.update(lanes_vehs[lane])

        dep_vehs = len(self.prev_vehs - current_vehs)
        arr_vehs = len(current_vehs - self.prev_vehs)
        self.dep_vehs_num.append(dep_vehs)
        self.arr_vehs_num.append(arr_vehs)
        self.prev_vehs = current_vehs

        
    def get_pressure(self, lanes_count):
        """
        Gets the pressure of the movement, the pressure is defined in traffic RL publications from PenState
        :param lanes_vehs: a dictionary with lane ids as keys and number of vehicles as values
        :returns: the pressure of the movement
        """
        pressure = 0
        if self.in_lanes:
            pressure += np.sum([lanes_count[x] / self.in_length for x in self.in_lanes])
        if self.out_lanes:
            pressure -= np.mean([lanes_count[x] / self.out_length for x in self.out_lanes])

        self.pressure = pressure
        return self.pressure

    def get_demand(self, lanes_count):
        """
        Gets the demand of the incoming lanes of the movement 
        the demand is the sum of the vehicles on all incoming lanes
        :param lanes_vehs: a dictionary with lane ids as keys and number of vehicles as values
        :returns: the demand of the movement
        """
        demand = int(np.sum([lanes_count[x] for x in self.in_lanes]))
        return demand

    def get_green_time(self, time, current_movements, eng):
        """
        Gets the predicted green time needed to clear the movement 
        :param time: the current timestep
        :param current_movements: a list of movements that are currently enabled
        :returns: the predicted green time of the movement
        """
        self.arr_rate = self.get_arr_veh_num(0, time) / time
        dep = self.get_dep_veh_num(0, time)

        green_time = 0
        LHS = dep + self.max_saturation * green_time

        clearing_time = self.clearing_time
            
        end_time = time + clearing_time + green_time - self.pass_time
        
        RHS = self.arr_rate * end_time
            
        while (RHS - LHS) > 0.1 and LHS < RHS:
            green_time += 1
            
            LHS = dep + self.max_saturation * green_time
            end_time = time + clearing_time + green_time - self.pass_time

            RHS = self.arr_rate * end_time

        return green_time
    
class Phase:
    """
    The class defining a Phase on an intersection, a Phase is defined by Movements which are enabled by it (given the green light)
    """
    def __init__(self, ID="", movements=[]):
        """
        initialises the Phase
        :param ID: the unique (for a given intersection) ID associated with the Phase, used by the engine to set phase on
        :param movements: the indeces of Movements which the Phase enables
        """
        self.ID = ID

        self.movements = movements
        self.vector = None
        self.to_action = None
        

class Lane:
    def __init__(self, eng, ID=""):
        self.ID = ID

        self.dep_vehs_num = []
        self.arr_vehs_num = []
        self.prev_vehs = set()
        self.speeds = []

        self.length = eng.get_lane_length(self.ID)
        

    def update_speeds(self, environ, veh_ids, speeds):
        self.speeds.append([speeds[id] for id in veh_ids if 'shadow' not in id])


    def update_flow_data(self, eng, lanes_vehs):
        """
        Updates the list containing the number vehicles that arrived and departed
        :param lanes_vehs: a dictionary with lane ids as keys and number of vehicles as values
        """
        current_vehs = set()
        current_vehs.update(lanes_vehs[self.ID])

        dep_vehs = len(self.prev_vehs - current_vehs)
        arr_vehs = len(current_vehs - self.prev_vehs)
        self.dep_vehs_num.append(dep_vehs)
        self.arr_vehs_num.append(arr_vehs)        
        self.prev_vehs = current_vehs

