#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import glob
import itertools
import os
import pickle
import numpy as np
import matplotlib as mpl

from datetime import datetime

plt.rcParams.update({'font.size': 7})


base_path = "../runs"

# scenarios = ['hangzhou', 'ny16']
scenarios = ['hangzhou_1', 'hangzhou_2']
# vote_scenarios = ['bipolar', 'balanced_mild', 'majority_mild', 'majority_extreme']
vote_scenarios = ['bipolar', 'random', 'majority_extreme']
# vote_scenarios = ['bipolar', 'majority_extreme']
vote_inputs = ['binary', 'cumulative']
vote_types = ["proportional", "majority"]

all_sats = pd.DataFrame(columns=["Satisfaction", "VoteScenario", "Aggregation", "VoteInput"])

for group in ["A","B"]:
    for scenario in scenarios:
        for vote_scenario in vote_scenarios:
            for aggregation in vote_types:
                for vote_input in vote_inputs:
                    for override in [False]:
                        if (aggregation=='majority') and (vote_input=='cumulative'):
                            continue
                        _path = f"{base_path}/{aggregation}/{vote_input}{'_override' if override else ''}/{scenario}_{vote_scenario}"

                        for trial, path in enumerate(glob.glob(f'{_path}*')):
                            satisfaction_path = path + "/alignment_drivers.pickle"
                            if os.path.isfile(satisfaction_path):
                                with open(satisfaction_path, "rb") as f:
                                    _sats = pickle.load(f)
                                    if (group is not None) and (len(_sats.keys())>1):
                                        sats = [i for i in _sats[group] if ~np.isnan(i)]
                                    else:
                                        sats = [i for x in _sats.values() for i in x] # dictionary of two groups
                                        sats = [x for x in sats if ~np.isnan(x)] # handle nan votes
                                    sats_df = pd.DataFrame(sats, columns=["Satisfaction"])
                                    sats_df["Scenario"] = scenario
                                    sats_df["VoteScenario"] = vote_scenario
                                    sats_df["Aggregation"] = aggregation
                                    sats_df["VoteInput"] = vote_input
                                    sats_df["VoteCombination"] = vote_input + " + " + aggregation
                                    sats_df['group'] = group
                                    if override:
                                        sats_df["VoteCombination"] = 'pure wait'
                                    all_sats = pd.concat([all_sats, sats_df]).reset_index(drop=True)
                            break
    print(all_sats.head())


    timestamp = datetime.now().strftime('%m%d_%H%M')
    plots_directory = f"../figs/alignment_distribution"
    os.makedirs(plots_directory, exist_ok=True)

    overlap = 1

all_sats.to_csv('alignments.csv', index=False)

# LOAD FROM HERE
all_sats = pd.read_csv('alignments.csv')

all_sats = all_sats.dropna(subset=['Satisfaction'])
all_sats=all_sats.fillna('None')
all_sats.group = all_sats.group.apply(lambda x: mapgroup[x])
all_sats.loc[all_sats.VoteScenario=='majority_extreme','VoteScenario'] = 'committed minority'

plt.rc('axes', facecolor=(0, 0, 0, 0), linewidth=0.7)
plt.rc('xtick', labelsize=7) 
plt.rc('ytick', labelsize=7)
plt.rc('font', size=8)
plt.rc('axes', titlesize=9, labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8, labelsize=8)


hatches = ['', '//']

row_order = ["binary + majority", 
             "binary + proportional",
             "cumulative + proportional"]


voting_scenarios = all_sats.VoteScenario.unique()

colors = [plt.cm.tab20.colors[i:i + 2] for i in range(0, len(voting_scenarios) * 2, 2)]
groups = ['stops','wait']

for key, grp in all_sats.groupby('Scenario'):
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(7, 3), sharex=True, layout='tight'
                            )

    height=3; aspect=1

    # handles = []
    for j, row in enumerate(row_order):
        for i, vs in enumerate(voting_scenarios):
            ax = axes[j, i]
            data = grp.loc[(grp.VoteScenario==vs) & (grp.VoteCombination==row)]
            sns.kdeplot(data=data,
                        x='Satisfaction', hue='group', hue_order=groups, alpha=1,
                            multiple='stack',  fill=True,
                        palette=colors[j], legend=False, clip=(0,1),
                            ax=ax)
            ax.set_yticks([])
            ax.set_ylabel(None)
            
            if i==0:
                _row = row.replace(" + ", " +\n ")
                txt = ax.text(-0.0, .2, _row, color='black', fontsize=8,
                            ha="right", va="center", transform=ax.transAxes)
                txt.set_in_layout(False)
                
            if j==0:
                ax.set_title(vs)
            for collection, hatch in zip(ax.collections[::-1], hatches * len(voting_scenarios)):
                collection.set_hatch(hatch)

    # construct proxy artist patches
    leg_artists = []
    for k, hatch in enumerate(hatches):
        p = mpl.patches.Patch(facecolor=f'{0.4+0.3*k}', hatch=hatch)
        # can also explicitly declare 2nd color like this
        #p = matplotlib.patches.Patch(facecolor='#DCDCDC', hatch=hatch_dict[i], edgecolor='0.5')

        leg_artists.append(p)

    # remove left spine
    sns.despine(left=True)
    
    # and add them to legend.
    leg = fig.legend(leg_artists, groups, loc='upper right', bbox_to_anchor=(0.1,1), ncols=2, 
               title='preference', facecolor='white')
    # leg.set_in_layout(False)
    
    fig.tight_layout()


    # adjust subplots to create overlap
    fig.subplots_adjust(hspace=-.2)
    
    fig.savefig(f'../figs/alignment_distribution/{key}_ridge.pdf', bbox_inches='tight')