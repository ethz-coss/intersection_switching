import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from run_vote import vote_scenarios, pure_methods, baseline
from radar_plot2 import ComplexRadar, format_cfg
import itertools
import pickle
import utils

pref_types = ['speed', 'stops', 'wait']

vote_scenarios = list(vote_scenarios.keys())
vote_inputs = ['binary','cumulative']

vote_types = ["proportional", "majority"]

# categories = ['Speed', 'Number of Stops', 'Wait Time']
# categories = [*categories, categories[0]]

scenarios = ['hangzhou_1','hangzhou_2']
all_data = {}
all_names = {}

def process_data(path):
    speeds_path = path + "/veh_speed_hist.pickle"
    stops_path = path + "/veh_stops.pickle"
    wait_path = path + "/veh_wait_time.pickle"

    with open(speeds_path, "rb") as f:
        speeds = pickle.load(f)

    veh_speed = [np.mean(speeds[x]) for x in speeds.keys()]

    with open(stops_path, "rb") as f:
        stops = pickle.load(f)

    veh_stops = [np.mean(stops[x]) for x in stops.keys()]

    with open(wait_path, "rb") as f:
        wait = pickle.load(f)

    wait = {k:v if v else [0] for k,v in wait.items()}

    veh_wait = [np.mean(wait[x]) for x in wait.keys()]

    return veh_wait, veh_stops, veh_speed

def rescale(d, ranges):
    _ranges = [*ranges, ranges[0]]
    new_data = []
    for i, _d in enumerate(d):
        minmax = _ranges[i]
        val = (_d-minmax[0])/(minmax[1]-minmax[0])
        new_data.append(val)
    return new_data

data_dict = []
for (scenario, vote_scenario) in itertools.product(scenarios, vote_scenarios):
    if vote_scenario in pure_methods+baseline:
        continue
    data = []
    names = []
    for vote_type in vote_types:
        for vote_input in vote_inputs:
                vote = vote_scenario
                avg_total_waits = []
                avg_total_stops = []
                avg_total_speeds = []
                for i in range(10):
                    if type(vote)==list:
                        path = f"../runs/{vote_type}/{vote_input}/{scenario}_{'_'.join(map(str,vote))}"
                    else:
                        path = f"../runs/{vote_type}/{vote_input}/{scenario}_{vote}"
                    # if i!=0:
                    path += f"({i})"

                    try:
                        veh_wait, veh_stops, veh_speed = process_data(path)
                    except FileNotFoundError:
                        print('FILE MISSING:', path)
                        continue

                    avg_wait, avg_stops, avg_speed = list(map(np.mean, [veh_wait, veh_stops, veh_speed]))
                    avg_total_waits.append(avg_wait)
                    avg_total_stops.append(avg_stops)
                    avg_total_speeds.append(avg_speed)


                    data_dict.append({
                        'speeds': avg_speed,
                        'stops': avg_stops,
                        'wait': avg_wait,
                        'raw_speeds': veh_speed,
                        'raw_stops': veh_stops,
                        'raw_wait': veh_wait,
                        'vote_type': vote_type,
                        'vote_input': vote_input,
                        'trial': i,
                        'name': f"{vote_input} + {vote_type}",
                        'vote_scenario':vote_scenario,
                        'scenario': scenario
                    })

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
                name = f"{vote_input} + {vote_type}"
                names.append(name)
                
    key = f"{scenario}_{vote_scenario}"
    all_data.update({key:data})
    all_names.update({key:names})



# with open('all_names.pickle', 'wb') as handle:
#     pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('all_data.pickle', 'wb') as handle:
#     pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Plotting happens here

# with open('all_names.pickle', 'rb') as handle:
#     all_names = pickle.load(handle)
# with open('all_data.pickle', 'rb') as handle:
#     all_data = pickle.load(handle)


for scenario in scenarios:
    data = []
    names = []
    for method in pure_methods+baseline:
        prefix = ''
        if method in pure_methods:
            prefix = 'pure'
            num = 10
        elif method in baseline:
            prefix = 'baseline'
            num = 10

        avg_total_waits = []
        avg_total_stops = []
        avg_total_speeds = []
        for i in range(num):
            path = f"../runs/{prefix}/{method}/{scenario}_{method}({i})"

            try:
                veh_wait, veh_stops, veh_speed = process_data(path)
            except FileNotFoundError:
                print('BASELINE FILES MISSING:', path)
                continue

            avg_wait, avg_stops, avg_speed = list(map(np.mean, [veh_wait, veh_stops, veh_speed]))
            avg_total_waits.append(avg_wait)
            avg_total_stops.append(avg_stops)
            avg_total_speeds.append(avg_speed)
            data_dict.append({
                'speeds': avg_speed,
                'stops': avg_stops,
                'wait': avg_wait,
                'raw_speeds': veh_speed,
                'raw_stops': veh_stops,
                'raw_wait': veh_wait,
                'vote_type': None,
                'vote_input': None,
                'trial': i,
                'name': f"{method}",
                'vote_scenario':None,
                'scenario': scenario
            })

        result = [np.mean(avg_total_speeds), np.mean(avg_total_stops), np.mean(avg_total_waits)]
        result = [*result, result[0]]
        data.append(result)

        name = f"{method}"
        names.append(name)
                
    key = f"{scenario}_pure"
    all_data.update({key:data})
    all_names.update({key:names})



plt.rcParams.update({'font.size': 6})

b1,b2, b3 = 0,0,0
colors = sns.color_palette('tab10', 10)
names_color_map = {'binary + majority': colors[0],
                   'binary + proportional': colors[1],
                   'cumulative + proportional': colors[2]
                    }


pure_fig = plt.figure(figsize=(4,2.5), dpi=300)
pure_subfigs = pure_fig.subfigures(1, 2)


for_df = []
for i, scenario in enumerate(scenarios):

    fig = plt.figure(figsize=(6,2.5), dpi=300)
    subfigs = fig.subfigures(1, sum([0 if vote_scenario in pure_methods+baseline else 1 for vote_scenario in vote_scenarios]))
    
    ## set bounds
    scenario_key_filter = [key for key in all_data.keys() if scenario in key]
    ranges = [(0, max([x[0] for key_prop in scenario_key_filter for x in all_data[key_prop]])),
            (max([x[1] for key_prop in scenario_key_filter for x in all_data[key_prop]]), 0),
            (max([x[2] for key_prop in scenario_key_filter for x in all_data[key_prop]]), 0)
            ]

    secranges = [(min([x[0] for key_prop in scenario_key_filter for x in all_data[key_prop]]), max([x[0] for key_prop in scenario_key_filter for x in all_data[key_prop]])),
            (max([x[1] for key_prop in scenario_key_filter for x in all_data[key_prop]]), min([x[1] for key_prop in scenario_key_filter for x in all_data[key_prop]])),
            (max([x[2] for key_prop in scenario_key_filter for x in all_data[key_prop]]), min([x[2] for key_prop in scenario_key_filter for x in all_data[key_prop]]))
            ]
    secranges = ranges

    # rr = [(0,1)]*3


    variables = ('Speed', 'Stops', 'Wait Time')

    key = f"{scenario}_pure"
        
    data = all_data[key]
    names = all_names[key]

    __purefig = pure_subfigs[i]

    # radar = ComplexRadar(axes, variables, ranges)
    pure_radar = ComplexRadar(__purefig, variables, ranges, format_cfg=format_cfg)

    sc_title = scenario
    if scenario == 'hangzhou':
        sc_title = "Hangzhou"
    elif scenario == 'ny16':
        sc_title = "NY16"

    for k, (d, name) in enumerate(zip(data, names)):
        print(d)
        dd = rescale(d, secranges)
        print(d, ranges)
        for_df.append({'area': sum(dd[:-1]),
                        'method': f'pure_{name}',
                        'scenario': sc_title})
        pure_radar.plot(d, label=utils.NAME_MAPPER.get(name, name), color=colors[k+5])
        pure_radar.fill(d, alpha=0.1, color=colors[k+5])


    __purefig.suptitle(sc_title, y=1.05, fontsize=8, fontweight='bold')

    save_name = f"../figs/{scenario}_metrics.pdf"
    print(save_name)
    for j, vote_scenario in enumerate(vote_scenarios):
        if vote_scenario in pure_methods+baseline:
            continue
        key = f"{scenario}_{vote_scenario}"
            
        data = all_data[key]
        names = all_names[key]

        fig1 = subfigs[j]

        # radar = ComplexRadar(axes, variables, ranges)
        radar = ComplexRadar(fig1, variables, ranges, format_cfg=format_cfg)

        if vote_scenario=='majority_extreme':
            vote_scenario = 'committed minority'
        for k, (d, name) in enumerate(zip(data, names)):
            if name=="cumulative + majority":
                continue
            kwargs = {}
            if 'cumulative' in name:
                 kwargs['zorder'] = 10
                 kwargs['ls'] = '--'
            dd = rescale(d, secranges)
            for_df.append({'area': sum(dd[:-1]),
                            'method': name,
                            'scenario': sc_title,
                            'vote_scenario': vote_scenario})
            radar.plot(d, label=name, color=names_color_map[name], **kwargs)
            radar.fill(d, alpha=0.1, color=names_color_map[name])

        fig1.suptitle(vote_scenario, y=1.05, fontsize=8, fontweight='bold')

    h,l = radar.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3)
    fig.tight_layout()
    fig.savefig(save_name, format='pdf', bbox_inches='tight')


h,l = pure_radar.get_legend_handles_labels()
pure_fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3)
pure_fig.savefig('../figs/pure.pdf', format='pdf', bbox_inches='tight')


import pandas as pd


hue_order = ["binary + majority", 
             "binary + proportional",
             "cumulative + proportional"]
df = pd.DataFrame(for_df)
g = sns.FacetGrid(df.dropna(), col='scenario', hue_order=hue_order, palette=names_color_map)
g.map_dataframe(sns.boxplot, y='area',
                x='vote_scenario', hue='method', hue_order=hue_order, palette=names_color_map)

pure_df = df[df.isnull().any(axis=1)].reset_index()
for i, (key, group) in enumerate(pure_df.groupby('scenario')):
    for j, (_, row) in enumerate(group.iterrows()):
        g.axes[0,i].axhline(row.area, color=colors[j+5], ls='--')
        g.axes[0,i].text(0.99,row.area, row.method, color='k', ha='right', va='bottom',# rotation=90,
            transform=g.axes[0,i].get_yaxis_transform())

h,l = g.axes[0,i].get_legend_handles_labels()
g.fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncols=3)
g.set_axis_labels("voting scenario", "overall system performance")

# g.fig.savefig('../figs/scalar_plot.pdf', bbox_inches='tight')