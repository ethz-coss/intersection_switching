import json

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


def flow_creator(dir, n_vehs=[11,5], routes=[['road_1','road_2'], ['road_3','road_4']], 
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

    print(f'{dir}/flow.json')
    with open(f'{dir}/flow.json', 'w') as f:
        f.write(json.dumps(flow_list, indent=2))
    return