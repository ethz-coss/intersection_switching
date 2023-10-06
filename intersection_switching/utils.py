import json
import os
import numpy as np

DEFAULT_VEHICLE = {
      "length": 5.0,
      "width": 2.0,
      "maxPosAcc": 2.0,
      "maxNegAcc": 4.5,
      "usualPosAcc": 2.0,
      "usualNegAcc": 4.5,
      "minGap": 2.5,
      "maxSpeed": 11.11,
      "headwayTime": 1.5
    }

NAME_MAPPER = {
    "localwait": "wait",
    "unique_stops": "stops"
}

def div0(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
def config_creator(dir, n_vehs, reward='speed', logpath=None):
    dir = os.path.abspath(dir)
    
    config =  {
        "interval": 1,
        "seed": 0,
        "dir": f'{dir}/',
        "roadnetFile": "roadnet.json",
        "flowFile": f"flow_{'_'.join(map(str, n_vehs))}_{reward}.json",
        "rlTrafficLight": True,
        "saveReplay": True,
        "roadnetLogFile": f"frontend/test_sphere.json",
        "replayLogFile": f"frontend/replay_file.txt",
        "laneChange": True
    }
    sim_config = f"{dir}/rings_{'_'.join(map(str, n_vehs))}_{reward}.config"
    with open(sim_config, 'w') as f:
        f.write(json.dumps(config, indent=2))
    return sim_config


def flow_creator(dir, n_vehs=[11,5], reward='speed', routes=[['road_1','road_2'], ['road_3','road_4']], 
                 loops=200, vehicle_params=DEFAULT_VEHICLE.copy()):
    flow_list = []
    for n_veh, route in zip(n_vehs, routes):
        route *= loops
        params = {
            'vehicle': vehicle_params,
            'interval': 1,
            'startTime': 0,
            'endTime': n_veh-1,
            'route': route
        }
        flow_list.append(params)

    with open(f"{dir}/flow_{'_'.join(map(str, n_vehs))}_{reward}.json", 'w') as f:
        f.write(json.dumps(flow_list, indent=2))
    return
