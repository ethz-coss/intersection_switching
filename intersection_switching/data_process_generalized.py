import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from run_vote import scenarios
from radar_plot2 import ComplexRadar, format_cfg
import itertools
import pickle


traffic_conditions = [[-1,-1]]

pref_types = ['speed', 'stops', 'wait']

vote_speed = [1.0, 0.0, 0.0]
vote_stops = [0.0, 1.0, 0.0]
vote_wait = [0.0, 0.0, 1.0]

vote_uniform_1 = [0.5, 0.5, 0.0]
vote_uniform_2 = [0.5, 0.0, 0.5]
vote_uniform_3 = [0.0, 0.5, 0.5]

vote_quarter_1 = [0.75, 0.25, 0.0]
vote_quarter_2 = [0.75, 0.0, 0.25]
vote_quarter_3 = [0.25, 0.0, 0.75]
vote_quarter_4 = [0.0, 0.25, 0.75]
vote_quarter_5 = [0.25, 0.75, 0.0]
vote_quarter_6 = [0.0, 0.75, 0.25]

# vote_types = [vote_speed, vote_stops, vote_wait, vote_uniform_1, vote_uniform_2, vote_uniform_3]#, vote_quarter_1, vote_quarter_2, vote_quarter_3, vote_quarter_4, vote_quarter_5, vote_quarter_6]

# vote_types = [vote_uniform_1, vote_uniform_2, vote_uniform_3]
# vote_types = [vote_stops, vote_wait, vote_uniform_3]
vote_scenarios = list(scenarios.keys())
vote_inputs = ['binary','cumulative']

vote_types = ["proportional", "majority"]

# categories = ['Speed', 'Number of Stops', 'Wait Time']
# categories = [*categories, categories[0]]

scenarios = ['hangzhou','ny16']
all_data = {}
all_names = {}


for (scenario, vote_scenario) in itertools.product(scenarios, vote_scenarios):
    data = []
    names = []
    for vote_type in vote_types:
        for vote_input in vote_inputs:
            for j, traffic in enumerate(traffic_conditions):

                vote = vote_scenario
                avg_total_waits = []
                avg_total_stops = []
                avg_total_speeds = []
                for i in range(5):
                    if type(vote)==list:
                        path = f"../runs/{vote_type}/{vote_input}/{scenario}_{'_'.join(map(str,vote))}"
                    else:
                        path = f"../runs/{vote_type}/{vote_input}/{scenario}_{vote}"
                    if i!=0:
                        path += f"({i})"
                    speeds_path = path + "/veh_speed_hist.pickle"
                    stops_path = path + "/veh_stops.pickle"
                    wait_path = path + "/veh_wait_time.pickle"

                    try:
                        with open(speeds_path, "rb") as f:
                            speeds = pickle.load(f)
                    except FileNotFoundError:
                        print('FILE MISSING:', speeds_path)
                        continue
                    avg_speed = np.mean([np.mean(speeds[x]) for x in speeds.keys()])
                    var_speed = np.std(([np.mean(speeds[x]) for x in speeds.keys()]))
                    # print("speed: ", avg_speed, var_speed)

                    with open(stops_path, "rb") as f:
                        stops = pickle.load(f)

                    avg_stops = np.mean([np.mean(stops[x]) for x in stops.keys()])
                    var_stops = np.std(([np.mean(stops[x]) for x in stops.keys()]))
                    # print("stops: ", avg_stops, var_stops)

                    with open(wait_path, "rb") as f:
                        wait = pickle.load(f)

                    wait = {k:v if v else [0] for k,v in wait.items()}

                    avg_wait = np.mean([np.mean(wait[x]) for x in wait.keys()])
                    var_wait = np.std(([np.mean(wait[x]) for x in wait.keys()]))
                    # print("wait: ", avg_wait, var_wait)

                    avg_total_waits.append(avg_wait)
                    avg_total_stops.append(avg_stops)
                    avg_total_speeds.append(avg_speed)


                result = [np.mean(avg_total_speeds), np.mean(avg_total_stops), np.mean(avg_total_waits)]
                result = [*result, result[0]]
                data.append(result)

                # if type(vote)==list:
                #     if vote == [0.0, 1.0, 0.0]:
                #         name = "Stops"
                #     elif vote == [0.0, 0.0, 1.0]:
                #         name = "Wait Times"
                # else:
                #     name = vote
                name = f"{vote_input}_{vote_type}"
                names.append(name)
                
    key = f"{scenario}_{vote_scenario}"
    all_data.update({key:data})
    all_names.update({key:names})



# with open('all_names.pickle', 'wb') as handle:
#     pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('all_data.pickle', 'wb') as handle:
#     pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Plotting happens here

# with open('all_names.pickle', 'rb') as handle:
#     all_names = pickle.load(handle)
# with open('all_data.pickle', 'rb') as handle:
#     all_data = pickle.load(handle)

plt.rcParams.update({'font.size': 6})

b1,b2, b3 = 0,0,0
colors = sns.color_palette(None, 7)
names_color_map = {'Stops': colors[1],
                    'Wait Times': colors[2],
                    'stops': colors[1],
                    'waits': colors[2],
                    'Prop S+W': colors[0],
                    # 'Major S+W': colors[6],
                    'bipolar': colors[3],
                    'random': colors[4],
                    'majority_extreme': colors[5]
                    }

for (scenario, vote_scenario) in itertools.product(scenarios, vote_scenarios):

        ## set bounds
        scenario_key_filter = [key for key in all_data.keys() if scenario in key]
        ranges = [(0, max([x[0] for key_prop in scenario_key_filter for x in all_data[key_prop]])),
                (max([x[1] for key_prop in scenario_key_filter for x in all_data[key_prop]]), 0),
                (max([x[2] for key_prop in scenario_key_filter for x in all_data[key_prop]]), 0)
                ]
        

        key_prop = f"{scenario}_{vote_scenario}"
        key = key_prop

            
        data = all_data[key]
        names = all_names[key]

        save_name = f"../figs/{scenario}_{vote_scenario}.pdf"
        print(save_name)
        # save_name = f"../figs/figure1.pdf"

        # fig1, axes = plt.subplots(1,1, subplot_kw={'projection':'polar'})
        # fig1 = plt.figure(figsize=(1.8, 1.8), dpi=300)
        fig1 = plt.figure(figsize=(3,3), dpi=300)

        variables = ('Speed', 'Stops', 'Wait Time')
        # radar = ComplexRadar(axes, variables, ranges)
        radar = ComplexRadar(fig1, variables, ranges, format_cfg=format_cfg)

        for i, (d, name) in enumerate(zip(data, names)):
            kwargs = {}
            if 'majority' in name:
                 kwargs['zorder'] = 10
                 kwargs['ls'] = '--'
            radar.plot(d, label=name, color=colors[i], **kwargs)
            radar.fill(d, alpha=0.1, color=colors[i])

        # name = "Major S+W"

        fig1.legend()
        if scenario==scenarios[0]:
            fig1.suptitle(vote_scenario, y=1.05, fontsize=8, fontweight='bold')
        if vote_scenario==vote_scenarios[0]:
            fig1.supylabel(scenario, fontsize=8, fontweight='bold')

        fig1.savefig(save_name, format='pdf', bbox_inches='tight')


