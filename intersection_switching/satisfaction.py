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

# vote_types = [vote_speed, vote_stops, vote_wait, vote_uniform_1, vote_uniform_2, vote_uniform_3]#, vote_quarter_1, vote_quarter_2, vote_quarter_3, vote_quarter_4, vote_quarter_5, vote_quarter_6]

# vote_types = [vote_uniform_1, vote_uniform_2, vote_uniform_3]
# vote_types = [vote_stops, vote_wait, vote_uniform_3]
vote_scenarios = list(scenarios.keys())
vote_inputs = ['binary','cumulative']

vote_types = ["proportional", "majority"]

scenarios = ['hangzhou','ny16']
all_data = {}
all_names = {}



# data = []
# names = []

# for traffic in traffic_conditions:
#     for vote_type in vote_types:

#         if type(vote)==list:
#             path = f"../runs/{vote_type}/{vote_input}/{scenario}_{'_'.join(map(str,vote))}"
#         else:
#             path = f"../runs/{vote_type}/{vote_input}/{scenario}_{vote}"

#         try:
#             path = f"proportional_100/{traffic[0]}_{traffic[1]}_{vote[0]}_{vote[1]}_{vote[2]}"
#             alignment_path = path + "/obj_alignment.pickle"
#             with open(alignment_path, "rb") as f:
#                 alignment = pickle.load(f)
#         except:
#             path = f"runs/{traffic[0]}_{traffic[1]}_{vote[0]}_{vote[1]}_{vote[2]}"
#             alignment_path = path + "/obj_alignment.pickle"
#             with open(alignment_path, "rb") as f:
#                 alignment = pickle.load(f)

#         align_df = pd.DataFrame(alignment)
#         align = align_df.iloc[:,:3].sub(align_df['reference'], axis=0).abs()
#         num = (1-align).sum(axis=0)
#         assert max(num)<=align.shape[0]
#         fractions = (1-align).sum(axis=0)/align.shape[0]
#         temp = fractions.to_dict()
#         temp.update({'vote': vote, 'traffic': traffic})
#         data.append(temp)
        

import itertools
import glob
import os
import pandas as pd

data = []

columns = ['stops','wait', 'reference']

# def satisfaction(row):
#     weights = row.vote_weights
#     sum_w = sum(weights.values())
#     satisfactions = []
#     for key in weights.keys():
#         if row[key]==row.reference: # vote is satisfied
#             satisfaction = weights[key]
#         else:
#             satisfaction = 0
#     #     row[f'satisfaction_{key}'] = satisfaction
#     # return row
#         satisfactions.append({'type': key,
#                               'value': 0 if not sum_w else satisfaction/sum_w,
#                               'scale': sum(weights.values())})
#     return satisfactions

def satisfaction(row):
    weights = row.vote_weights
    sum_w = sum(weights.values())
    satisfaction = 0
    for key in weights.keys():
        if row[key]==row.reference: # vote is satisfied
            satisfaction += weights[key]
    row['satisfaction'] = 0 if not sum_w else satisfaction/sum_w
    return row


correlate = lambda x,y: (x==y).sum()/len(x)
for (scenario, vote_scenario) in itertools.product(scenarios, vote_scenarios):
    for aggregation in vote_types:
        for vote_input in vote_inputs:

            _path = f"../runs/{aggregation}/{vote_input}/{scenario}_{vote_scenario}"
            if not os.path.isdir(_path):
                print('getting runs', _path)
                # _path = f"runs/{traffic[0]}_{traffic[1]}_{vote[0]}_{vote[1]}_{vote[2]}"
            else:
                print(len(glob.glob(f'{_path}*')))
            for trial, path in enumerate(glob.glob(f'{_path}*')[:5]):

                alignment_path = path + "/obj_alignment.pickle"
                with open(alignment_path, "rb") as f:
                    alignment = pickle.load(f)


                align_df = pd.DataFrame(alignment)


                # ss = []
                # for i, row in align_df.iterrows():
                #     ss.extend(satisfaction(row))
                ss = align_df.apply(lambda x: satisfaction(x), axis=1)

                # align_df[list(align_df.vote_weights[0].keys())] = zip(*align_df.apply(lambda x: satisfaction(x), axis=1))
                # _data = [{0: x,
                #           'level_0': corrs.columns[y],
                #           'level_1': corrs.columns[z]} for x,y,z in zip(unique, *upper_right_entries)]
                _data = pd.DataFrame(ss)
                _data['vote'] = vote_scenario
                _data['trial'] = trial
                _data['aggregation'] = aggregation
                _data['scenario'] = scenario
                _data['vote_input'] = vote_input

                data.append(_data)
        


newdf = pd.concat(data).reset_index(drop=True)
newdf['cross'] = newdf['vote_input'] + '_' + newdf['aggregation']

sns.set_context('paper')
# sns.color_palette('tab10')
g = sns.FacetGrid(newdf, col='scenario', margin_titles=True, height=2.2)
g.map_dataframe(sns.barplot, x='vote',
                y='satisfaction', hue='cross', palette='pastel')
g.add_legend(ncol=3, bbox_to_anchor=(0.53,1.03));
g.set(ylim=(0, 1), xlabel=None, ylabel='correlation')
# g.set_titles(template=None, row_template="aggregation: {row_name}", col_template="input: {col_name}")
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)
g.fig.savefig('../figs/alignment.pdf', bbox_inches='tight')

# for scenario in scenarios:
#     _df = newdf.query('scenario==@scenario')
#     sns.set_context('paper')
#     # sns.color_palette('tab10')
#     g = sns.FacetGrid(_df, col="vote_input", row='aggregation', margin_titles=True, height=2.2)
#     g.map_dataframe(sns.barplot, x='vote',
#                     y='value', hue='type', palette='pastel')
#     g.add_legend(ncol=3, bbox_to_anchor=(0.53,1.03));
#     g.set(ylim=(0, 1), xlabel=None, ylabel='correlation')
#     g.set_titles(template=None, row_template="aggregation: {row_name}", col_template="input: {col_name}")
#     for ax in g.axes.flat:
#         for label in ax.get_xticklabels():
#             label.set_rotation(90)
#     g.fig.savefig(f'../figs/alignment_{scenario}.pdf', bbox_inches='tight')