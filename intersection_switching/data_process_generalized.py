import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from radar_plot import ComplexRadar 

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
vote_types = [vote_stops, vote_wait, vote_uniform_3]
vote_modes = ["proportional", "majority"]
vote_modes = ["proportional"]

categories = ['Speed', 'Number of Stops', 'Wait Time']
categories = [*categories, categories[0]]

all_data = {}
all_names = {}

for vote_type in vote_modes:
    for j, traffic in enumerate(traffic_conditions):
        data = []
        names = []

        for vote in vote_types:
            avg_total_waits = []
            avg_total_stops = []
            avg_total_speeds = []
            for i in range(100):
                # if (vote == vote_stops or vote == vote_wait) and i >= 1:
                #     break

                if i == 0:
                    path = f"../runs/{vote_type}_100/{traffic[0]}_{traffic[1]}_{vote[0]}_{vote[1]}_{vote[2]}"
                else:
                    path = f"../runs/{vote_type}_100/{traffic[0]}_{traffic[1]}_{vote[0]}_{vote[1]}_{vote[2]}({i})"

                speeds_path = path + "/veh_speed_hist.pickle"
                stops_path = path + "/veh_stops.pickle"
                wait_path = path + "/veh_wait_time.pickle"

                with open(speeds_path, "rb") as f:
                    speeds = pickle.load(f)

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

                # if vote == [0.0, 0.5, 0.5]:
                #     print(traffic[0], traffic[1], "per car")
                #     print("wait: ", avg_wait, var_wait)
                #     print("stops: ", avg_stops, var_stops)

            # if vote == [0.0, 0.5, 0.5]:
            #     print(vote, traffic[0], traffic[1])
            #     print("wait: ", np.mean(avg_total_waits), np.std(avg_total_waits))
            #     print("stops: ", np.mean(avg_total_stops), np.std(avg_total_stops))
            #     print("speeds: ", np.mean(avg_total_speeds), np.std(avg_total_speeds))

            #     plt.scatter(avg_total_waits, avg_total_stops)
            #     plt.show()

            result = [np.mean(avg_total_speeds), np.mean(avg_total_stops), np.mean(avg_total_waits)]
            # print(np.mean(avg_total_speeds), np.var(avg_total_speeds),
            #       np.mean(avg_total_stops), np.var(avg_total_stops),
            #       np.mean(avg_total_waits), np.var(avg_total_waits))
            result = [*result, result[0]]
            data.append(result)

            if vote == [0.0, 1.0, 0.0]:
                name = "Stops"
            elif vote == [0.0, 0.0, 1.0]:
                name = "Wait Times"
            else:
                name = "Prop S+W"

            names.append(name)
        
        key = f"{traffic[0]}_{traffic[1]}_{vote_type}"
        all_data.update({key:data})
        all_names.update({key:names})

plt.rcParams.update({'font.size': 12})

b1,b2, b3 = 0,0,0


##### for uniform axis limits
# for traffic in traffic_conditions:

#     _labels = [f"{traffic[0]}_{traffic[1]}_proportional", f"{traffic[0]}_{traffic[1]}_majority"]

#     b1 = max(b1, max([x[0] for ll in _labels for x in all_data[ll] ]))
#     b2 = max(b2, max([x[1] for ll in _labels for x in all_data[ll] ]))
#     b3 = max(b3, max([x[2] for ll in _labels for x in all_data[ll] ]))
# ranges = [(0, b1),
#             (b2, 0),
#             (b3, 0)]

colors = sns.color_palette(None, 7)
names_color_map = {'Stops': colors[1],
                    'Wait Times': colors[2],
                    'Prop S+W': colors[0],
                    'Major S+W': colors[6]
                    }
for traffic in traffic_conditions:

    # key_prop = f"{traffic[0]}_{traffic[1]}_proportional"
    # key_major = f"{traffic[0]}_{traffic[1]}_majority"
    
    # ranges = [(0, max(max([x[0] for x in all_data[key_prop]]), max([x[0] for x in all_data[key_major]]))),
    #           (max(max([x[1] for x in all_data[key_prop]]), max([x[1] for x in all_data[key_major]])), 0),
    #           (max(max([x[2] for x in all_data[key_prop]]), max([x[2] for x in all_data[key_major]])), 0)
    #            ]

    key_prop = f"{traffic[0]}_{traffic[1]}_proportional"
    
    ranges = [(0, max([x[0] for x in all_data[key_prop]])),
              (max([x[1] for x in all_data[key_prop]]), 0),
              (max([x[2] for x in all_data[key_prop]]), 0)
               ]


    for vote_type in ["proportional"]:
        if vote_type == "majority":
            key = key_major
        else:
            key = key_prop
        
        data = all_data[key]
        names = all_names[key]

        save_name = f"../figs/{traffic[0]}_{traffic[1]}.pdf"
        print(save_name)
        # save_name = f"../figs/figure1.pdf"

        fig1, axes = plt.subplots(1,1, subplot_kw={'projection':'polar'})

        variables = ('Speed', 'Stops', 'Wait Time')
        radar = ComplexRadar(axes, variables, ranges)

        for d, name in zip(data, names):
            radar.plot(d, label=name, color=names_color_map[name])
            radar.fill(d, alpha=0.2, color=names_color_map[name])

        name = "Major S+W"
        # radar.plot(all_data[key_major][2], label=name, color=names_color_map[name])
        # radar.fill(all_data[key_major][2], alpha=0.2, color=names_color_map[name])

        fig1.legend()

        # if traffic[0] == 11:
        #     if traffic[1] == 11:
        #         axes.set_title("Low Balanced")
        #     else:
        #         axes.set_title("Low Unbalanced")
        # elif traffic[0] == 22:
        #     if traffic[1] == 22:
        #         axes.set_title("Medium Balanced")
        #     else:
        #         axes.set_title("Medium Unbalanced")
        # elif traffic[0] == 32:
        #     if traffic[1] == 32:
        #         axes.set_title("High Balanced")
        #     else:
        #         axes.set_title("High Unbalanced")



        fig1.savefig(save_name, format='pdf', bbox_inches='tight')


