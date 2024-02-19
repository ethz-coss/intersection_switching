import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from radar_plot import ComplexRadar 
import seaborn as sns

low_balanced = [11, 11]
low_unbalanced = [11, 6]

medium_balanced = [22, 22]
medium_unbalanced = [22, 11]

high_balanced = [32, 32]
high_unbalanced = [32, 16]


# traffic_conditions = [low_balanced, low_unbalanced, medium_balanced, medium_unbalanced, high_balanced, high_unbalanced]

traffic_dict = {('low', 'balanced'): [11,11],
                      ('low', 'unbalanced'): [11,6],
                      ('medium', 'balanced'): [22,22],
                      ('medium', 'unbalanced'): [22,11],
                      ('high', 'balanced'): [32,32],
                      ('high', 'unbalanced'): [32,16],
                     }

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

categories = ['Speed', 'Number of Stops', 'Wait Time']
categories = [*categories, categories[0]]
vote_types = [vote_uniform_3, ['both']]


# all_data = {}
# all_names = {}

# for j, (label, traffic) in enumerate(traffic_dict.items()):
#     data = []
#     names = []
#     for folder in ['proportional_100','both_lin_combo_test', 'both_cobb_doug_test']:
#         if 'both' in folder:
#             vote = ['both']
#         else:
#             vote = vote_uniform_3
#         avg_total_waits = []
#         avg_total_stops = []
#         avg_total_speeds = []


#         _path = f"../{folder}/{'_'.join(map(str,traffic))}_{'_'.join(map(str,vote))}"

#         # for trial, path in enumerate([_path]):
#         for trial, path in enumerate(glob.glob(f'{_path}*')):
#             print('globbing', path)
#             speeds_path = path + "/veh_speed_hist.pickle"
#             stops_path = path + "/veh_stops.pickle"
#             wait_path = path + "/veh_wait_time.pickle"

#             with open(speeds_path, "rb") as f:
#                 speeds = pickle.load(f)

#             avg_speed = np.mean([np.mean(speeds[x]) for x in speeds.keys()])
#             var_speed = np.std(([np.mean(speeds[x]) for x in speeds.keys()]))

#             with open(stops_path, "rb") as f:
#                 stops = pickle.load(f)

#             avg_stops = np.mean([np.mean(stops[x]) for x in stops.keys()])
#             var_stops = np.std(([np.mean(stops[x]) for x in stops.keys()]))

#             with open(wait_path, "rb") as f:
#                 wait = pickle.load(f)

#             wait = {k:v if v else [0] for k,v in wait.items()}

#             avg_wait = np.mean([np.mean(wait[x]) for x in wait.keys()])
#             var_wait = np.std(([np.mean(wait[x]) for x in wait.keys()]))

#             avg_total_waits.append(avg_wait)
#             avg_total_stops.append(avg_stops)
#             avg_total_speeds.append(avg_speed)

#         result = [np.mean(avg_total_speeds), np.mean(avg_total_stops), np.mean(avg_total_waits)]
#         result = [*result, result[0]]
#         data.append(result)

#         if vote == vote_uniform_3:
#             name = "Prop S+W"
#         elif vote == ['both']:
#             if folder == 'both_lin_combo_test':
#                 name = "Linear comb."
#             elif folder == 'both_cobb_doug_test':
#                 name = "Cobb Doug"

#         names.append(name)


#     key = f"{'_'.join(map(str,traffic))}"
#     all_data.update({key:data})
#     all_names.update({key:names})

#     variables = ('Speed', 'Stops', 'Wait Time')
#     ranges = [(0, max([x[0] for x in data])), (max([x[1] for x in data]), 0), (max([x[2] for x in data]), 0)]            
#     # plotting

    

# data = {'all_data': all_data, 
#         'all_names': all_names}

# with open('data.pickle', 'wb') as f:
#     pickle.dump(data, f)

with open('data.pickle', "rb") as f:
    data = pickle.load(f)
all_data = data['all_data']
all_names = data['all_names']

plt.rcParams.update({'font.size': 12})
b1,b2, b3 = 0,0,0

# for label, traffic in traffic_dict.items():

#     _labels = [f"{'_'.join(map(str,traffic))}"]

#     b1 = max(b1, max([x[0] for ll in _labels for x in all_data[ll] ]))
#     b2 = max(b2, max([x[1] for ll in _labels for x in all_data[ll] ]))
#     b3 = max(b3, max([x[2] for ll in _labels for x in all_data[ll] ]))
# ranges = [(0, b1),
#             (b2, 0),
#             (b3, 0)]
    

colors = sns.color_palette(None, 6)
names_color_map = {'Stops': colors[1],
                    'Wait Times': colors[2],
                    'Prop S+W': colors[0],
                    'Major S+W': colors[3],
                    'Linear comb.': colors[4],
                    'Cobb Doug': colors[5]
                    }


# for key, data in all_data.items():
for label, traffic in traffic_dict.items():
    _labels = f"{'_'.join(map(str,traffic))}"

    fig1, axes = plt.subplots(1,1, subplot_kw={'projection':'polar'})
    
    # for name in ["Prop S+W", "Linear comb.", "Cobb Doug"]:
    key = f"{'_'.join(map(str,traffic))}"
    data = all_data[key]
    names = all_names[key]


    b1 = max([x[0] for x in data ])
    b2 = max([x[1] for x in data ])
    b3 = max([x[2] for x in data ])
    ranges = [(0, b1),
                (b2, 0),
                (b3, 0)]

    save_name = f"../figs/{'_'.join(map(str,traffic))}_appendix.pdf"
    print(save_name, ranges)
    print(data)
    # save_name = f"../figs/figure1.pdf"


    variables = ('Speed', 'Stops', 'Wait Time')
    radar = ComplexRadar(axes, variables, ranges)
    for d, name in zip(data, names):
        labelname = name
        if 'Cobb' in name:
            labelname = "Cobb–Doug"
        radar.plot(d, label=labelname, color=names_color_map[name])
        radar.fill(d, alpha=0.2, color=names_color_map[name])

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
    axes.set_title(' '.join([x.capitalize() for x in label]))


    fig1.savefig(save_name, format='pdf', bbox_inches='tight')
    
    #####################

    
#     fig1, axes = plt.subplots(1,1, subplot_kw={'projection':'polar'})

#     radar = ComplexRadar(axes, variables, ranges)

#     for d, name in zip(data, names):
#         radar.plot(d, label=name)

#         radar.fill(d, alpha=0.2)
        
    
#     save_name = f"../figs/{traffic[0]}_{traffic[1]}.pdf"
#     # save_name = f"../figs/figure1.pdf"
#     fig1.legend()

#     if traffic[0] == 11:
#         if traffic[1] == 11:
#             axes.set_title("Low Balanced")
#         else:
#             axes.set_title("Low Unbalanced")
#     elif traffic[0] == 22:
#         if traffic[1] == 22:
#             axes.set_title("Medium Balanced")
#         else:
#             axes.set_title("Medium Unbalanced")
#     elif traffic[0] == 32:
#         if traffic[1] == 32:
#             axes.set_title("High Balanced")
#         else:
#             axes.set_title("High Unbalanced")

                    
        
#     fig1.savefig(save_name, format='pdf', bbox_inches='tight')


