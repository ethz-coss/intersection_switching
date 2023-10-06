#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import glob
import itertools
import os
import pickle
import joypy
import numpy as np
# from pandas import (DataFrame, Series)
# from pandas.core.groupby import DataFrameGroupBy
from joypy.joyplot import *
from joypy.joyplot import (_is_numeric, _remove_na, _x_range,
                           _subplots, _flatten, _setup_axis, _DEBUG)

from datetime import datetime

def joyplot(data, column=None, by=None, grid=False,
            xlabelsize=None, xrot=None, ylabelsize=None, yrot=None,
            ax=None, figsize=None,
            hist=False, bins=10,
            fade=False, ylim='max',
            fill=True, linecolor=None,
            overlap=1, background=None,
            labels=None, xlabels=True, ylabels=True,
            range_style='all',
            x_range=None,
            title=None,
            colormap=None,
            color=None,
            normalize=True,
            floc=None,
            medians=False,
            **kwds):
    """
    Draw joyplot of a DataFrame, or appropriately nested collection,
    using matplotlib and pandas.

    A joyplot is a stack of vertically aligned density plots / histograms.
    By default, if 'data' is a DataFrame,
    this function will plot a density plot for each column.

    This wrapper method tries to convert whatever structure is given
    to a nested collection of lists with additional information
    on labels, and use the private _joyplot function to actually
    draw theh plot.

    Parameters
    ----------
    data : DataFrame, Series or nested collection
    column : string or sequence
        If passed, will be used to limit data to a subset of columns
    by : object, optional
        If passed, used to form separate plot groups
    grid : boolean, default True
        Whether to show axis grid lines
    labels : boolean or list, default True.
        If list, must be the same size of the de
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    figsize : tuple
        The size of the figure to create in inches by default
    hist : boolean, default False
    bins : integer, default 10
        Number of histogram bins to be used
    color : color or colors to be used in the plots. It can be:
        a string or anything interpretable as color by matplotib;
        a list of colors. See docs / examples for more details.
    kwds : other plotting keyword arguments
        To be passed to hist/kde plot function
    """

    if column is not None:
        if not isinstance(column, (list, np.ndarray)):
            column = [column]

    def _grouped_df_to_standard(grouped, column):
        converted = []
        labels = []
        for i, (key, group) in enumerate(grouped):
            if column is not None:
                group = group[column]
            labels.append(key)
            converted.append([_remove_na(group[c]) for c in group.columns if _is_numeric(group[c])])
            if i == 0:
                sublabels = [col for col in group.columns if _is_numeric(group[col])]
        return converted, labels, sublabels

    #################################################################
    # GROUPED
    # - given a grouped DataFrame, a group by key, or a dict of dicts of Series/lists/arrays
    # - select the required columns/Series/lists/arrays
    # - convert to standard format: list of lists of non-null arrays
    #   + extra parameters (labels and sublabels)
    #################################################################
    if isinstance(data, DataFrameGroupBy):
        grouped = data
        converted, _labels, sublabels = _grouped_df_to_standard(grouped, column)
        if labels is None:
            labels = _labels
    elif by is not None and isinstance(data, DataFrame):
        grouped = data.groupby(by)
        if column is None:
            # Remove the groupby key. It's not automatically removed by pandas.
            column = list(data.columns)
            column.remove(by)
        converted, _labels, sublabels = _grouped_df_to_standard(grouped, column)
        if labels is None:
            labels = _labels
        # If there is at least an element which is not a list of lists.. go on.
    elif isinstance(data, dict) and all(isinstance(g, dict) for g in data.values()):
        grouped = data
        if labels is None:
            labels = list(grouped.keys())
        converted = []
        for i, (key, group) in enumerate(grouped.items()):
            if column is not None:
                converted.append([_remove_na(g) for k,g in group.items() if _is_numeric(g) and k in column])
                if i == 0:
                    sublabels = [k for k,g in group.items() if _is_numeric(g)]
            else:
                converted.append([_remove_na(g) for k,g in group.items() if _is_numeric(g)])
                if i == 0:
                    sublabels = [k for k,g in group.items() if _is_numeric(g)]
    #################################################################
    # PLAIN:
    # - given a DataFrame or list/dict of Series/lists/arrays
    # - select the required columns/Series/lists/arrays
    # - convert to standard format: list of lists of non-null arrays + extra parameter (labels)
    #################################################################
    elif isinstance(data, DataFrame):
        if column is not None:
            data = data[column]
        converted = [[_remove_na(data[col])] for col in data.columns if _is_numeric(data[col])]
        labels = [col for col in data.columns if _is_numeric(data[col])]
        sublabels = None
    elif isinstance(data, dict):
        if column is not None:
            converted = [[_remove_na(g)] for k,g in data.items() if _is_numeric(g) and k in column]
            labels = [k for k,g in data.items() if _is_numeric(g) and k in column]
        else:
            converted = [[_remove_na(g)] for k,g in data.items() if _is_numeric(g)]
            labels = [k for k,g in data.items() if _is_numeric(g)]
        sublabels = None
    elif isinstance(data, list):
        if column is not None:
            converted = [[_remove_na(g)] for g in data if _is_numeric(g) and i in column]
        else:
            converted = [[_remove_na(g)] for g in data if _is_numeric(g)]
        if labels and len(labels) != len(converted):
            raise ValueError("The number of labels does not match the length of the list.")

        sublabels = None
    else:
        raise TypeError("Unknown type for 'data': {!r}".format(type(data)))

    if ylabels is False:
        labels = None

    if all(len(subg)==0 for g in converted for subg in g):
        raise ValueError("No numeric values found. Joyplot requires at least a numeric column/group.")

    if any(len(subg)==0 for g in converted for subg in g):
        warn("At least a column/group has no numeric values.")


    return _joyplot(converted, labels=labels, sublabels=sublabels,
                    grid=grid,
                    xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot,
                    ax=ax, figsize=figsize,
                    hist=hist, bins=bins,
                    fade=fade, ylim=ylim,
                    fill=fill, linecolor=linecolor,
                    overlap=overlap, background=background,
                    xlabels=xlabels,
                    range_style=range_style, x_range=x_range,
                    title=title,
                    colormap=colormap,
                    color=color,
                    normalize=normalize,
                    floc=floc,
                    medians=medians,
                    **kwds)

def _joyplot(data,
             grid=False,
             labels=None, sublabels=None,
             xlabels=True,
             xlabelsize=None, xrot=None,
             ylabelsize=None, yrot=None,
             ax=None, figsize=None,
             hist=False, bins=10,
             fade=False,
             xlim=None, ylim='max',
             fill=True, linecolor=None,
             overlap=1, background=None,
             range_style='all', x_range=None, tails=0.2,
             title=None,
             legend=False, loc="upper right",
             colormap=None, color=None,
             normalize=True,
             floc=None,
             medians=False,
             **kwargs):
    """
    Internal method.
    Draw a joyplot from an appropriately nested collection of lists
    using matplotlib and pandas.

    Parameters
    ----------
    data : DataFrame, Series or nested collection
    grid : boolean, default True
        Whether to show axis grid lines
    labels : boolean or list, default True.
        If list, must be the same size of the de
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    figsize : tuple
        The size of the figure to create in inches by default
    hist : boolean, default False
    bins : integer, default 10
        Number of histogram bins to be used
    kwarg : other plotting keyword arguments
        To be passed to hist/kde plot function
    """

    if fill is True and linecolor is None:
        linecolor = "k"

    if sublabels is None:
        legend = False

    def _get_color(i, num_axes, j, num_subgroups):
        if isinstance(color, list):
            return color[j] if num_subgroups > 1 else color[i]
        elif color is not None:
            return color
        elif isinstance(colormap, list):
            return colormap[j](i/num_axes)
        elif color is None and colormap is None:
            num_cycle_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][j % num_cycle_colors]
        else:
            return colormap(i/num_axes)

    ygrid = (grid is True or grid == 'y' or grid == 'both')
    xgrid = (grid is True or grid == 'x' or grid == 'both')

    num_axes = len(data)

    if x_range is None:
        global_x_range = _x_range([v for g in data for sg in g for v in sg])
    else:
        global_x_range = _x_range(x_range, 0.0)
    global_x_min, global_x_max = min(global_x_range), max(global_x_range)

    # Each plot will have its own axis
    fig, axes = _subplots(naxes=num_axes, ax=ax, squeeze=False,
                          sharex=True, sharey=False, figsize=figsize,
                          layout_type='vertical')
    _axes = _flatten(axes)

    # The legend must be drawn in the last axis if we want it at the bottom.
    if loc in (3, 4, 8) or 'lower' in str(loc):
        legend_axis = num_axes - 1
    else:
        legend_axis = 0

    # A couple of simple checks.
    if labels is not None:
        assert len(labels) == num_axes
    if sublabels is not None:
        assert all(len(g) == len(sublabels) for g in data)
    if isinstance(color, list):
        assert all(len(g) <= len(color) for g in data)
    if isinstance(colormap, list):
        assert all(len(g) == len(colormap) for g in data)

    for i, group in enumerate(data):

        a = _axes[i]
        group_zorder = i
        if fade:
            kwargs['alpha'] = _get_alpha(i, num_axes)

        num_subgroups = len(group)

        if hist:
            # matplotlib hist() already handles multiple subgroups in a histogram
            a.hist(group, label=sublabels, bins=bins, color=color,
                   range=[min(global_x_range), max(global_x_range)],
                   edgecolor=linecolor, zorder=group_zorder, **kwargs)
        else:
            for j, subgroup in enumerate(group):

                # Compute the x_range of the current plot
                if range_style == 'all':
                # All plots have the same range
                    x_range = global_x_range
                elif range_style == 'own':
                # Each plot has its own range
                    x_range = _x_range(subgroup, tails)
                elif range_style == 'group':
                # Each plot has a range that covers the whole group
                    x_range = _x_range(group, tails)
                elif isinstance(range_style, (list, np.ndarray)):
                # All plots have exactly the range passed as argument
                    x_range = _x_range(range_style, 0.0)
                else:
                    raise NotImplementedError("Unrecognized range style.")

                if sublabels is None:
                    sublabel = None
                else:
                    sublabel = sublabels[j]

                element_zorder = group_zorder + j/(num_subgroups+1)
                element_color = _get_color(i, num_axes, j, num_subgroups)

                plot_density(a, x_range, subgroup,
                             fill=fill, linecolor=linecolor, label=sublabel,
                             zorder=element_zorder, color=element_color,
                             bins=bins, **kwargs)

                if medians:
                    _median = np.median(group)
                    a.axvline(_median, ls='--', c='0.3', zorder=3)

        # Setup the current axis: transparency, labels, spines.
        col_name = None if labels is None else labels[i]
        _setup_axis(a, global_x_range, col_name=col_name, grid=ygrid,
                ylabelsize=ylabelsize, yrot=yrot)

        # When needed, draw the legend
        if legend and i == legend_axis:
            a.legend(loc=loc)
            # Bypass alpha values, in case
            for p in a.get_legend().get_patches():
                p.set_facecolor(p.get_facecolor())
                p.set_alpha(1.0)
            for l in a.get_legend().get_lines():
                l.set_alpha(1.0)


    # Final adjustments

    # Set the y limit for the density plots.
    # Since the y range in the subplots can vary significantly,
    # different options are available.
    if ylim == 'max':
        # Set all yaxis limit to the same value (max range among all)
        max_ylim = max(a.get_ylim()[1] for a in _axes)
        min_ylim = min(a.get_ylim()[0] for a in _axes)
        for a in _axes:
            a.set_ylim([min_ylim - 0.1*(max_ylim-min_ylim), max_ylim])

    elif ylim == 'own':
        # Do nothing, each axis keeps its own ylim
        pass

    else:
        # Set all yaxis lim to the argument value ylim
        try:
            for a in _axes:
                a.set_ylim(ylim)
        except:
            print("Warning: the value of ylim must be either 'max', 'own', or a tuple of length 2. The value you provided has no effect.")

    # Compute a final axis, used to apply global settings
    last_axis = fig.add_subplot(1, 1, 1)

    # Background color
    if background is not None:
        last_axis.patch.set_facecolor(background)

    for side in ['top', 'bottom', 'left', 'right']:
        last_axis.spines[side].set_visible(_DEBUG)

    # This looks hacky, but all the axes share the x-axis,
    # so they have the same lims and ticks
    last_axis.set_xlim(_axes[0].get_xlim())
    if xlabels is True:
        last_axis.set_xticks(np.array(_axes[0].get_xticks()[1:-1]))
        for t in last_axis.get_xticklabels():
            t.set_visible(True)
            t.set_fontsize(xlabelsize)
            t.set_rotation(xrot)

        # If grid is enabled, do not allow xticks (they are ugly)
        if xgrid:
            last_axis.tick_params(axis='both', which='both',length=0)
    else:
        last_axis.xaxis.set_visible(False)

    last_axis.yaxis.set_visible(False)
    last_axis.grid(xgrid)


    # Last axis on the back
    last_axis.zorder = min(a.zorder for a in _axes) - 1
    _axes = list(_axes) + [last_axis]

    if title is not None:
        plt.title(title)


    # The magic overlap happens here.
    h_pad = 5 + (- 5*(1 + overlap))
    # fig.tight_layout(h_pad=h_pad)

    return fig, _axes

# joypy._joyplot = _joyplot


plt.rcParams.update({'font.size': 7})


base_path = "../runs"

# scenarios = ['hangzhou', 'ny16']
scenarios = ['hangzhou_1', 'hangzhou_2']
# vote_scenarios = ['bipolar', 'balanced_mild', 'majority_mild', 'majority_extreme']
vote_scenarios = ['bipolar', 'random', 'majority_extreme']
# vote_scenarios = ['bipolar', 'majority_extreme']
vote_inputs = ['binary', 'cumulative']
vote_types = ["proportional", "majority"]


for group in ["A","B"]:
    group = None ## Toggle to combine the ridge plots
    all_sats = pd.DataFrame(columns=["Satisfaction", "VoteScenario", "Aggregation", "VoteInput"])
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

                                    if group is not None:
                                        sats = _sats[group]
                                    else:
                                        sats = [i for x in _sats.values() for i in x] # dictionary of two groups
                                        sats = [x for x in sats if ~np.isnan(x)] # handle nan votes
                                    sats_df = pd.DataFrame(sats, columns=["Satisfaction"])
                                    sats_df["Scenario"] = scenario
                                    sats_df["VoteScenario"] = vote_scenario
                                    sats_df["Aggregation"] = aggregation
                                    sats_df["VoteInput"] = vote_input
                                    sats_df["VoteCombination"] = vote_input + " + " + aggregation
                                    if override:
                                        sats_df["VoteCombination"] = 'pure wait'

                                    # sats_df["Override"] = override

                                    all_sats = pd.concat([all_sats, sats_df])
                            break
    print(all_sats.head())


    timestamp = datetime.now().strftime('%m%d_%H%M')
    plots_directory = f"../figs/alignment_distribution"
    os.makedirs(plots_directory, exist_ok=True)

    overlap = 1



    for scenario in scenarios:
        fig = plt.figure(figsize=(6,4), dpi=300)
        subfigs = fig.subfigures(1, len(vote_scenarios))

        for i, vote_scenario in enumerate(vote_scenarios):
            axes = subfigs[i].subplots(all_sats["VoteCombination"].unique().size,1)
            axes = axes.flatten()

            use_ylabels = False
            if i==0:
                use_ylabels = True
            filtered_data = all_sats[(all_sats['VoteScenario'] == vote_scenario) & (all_sats['Scenario'] == scenario)]
            if vote_scenario=='majority_extreme':
                vote_scenario = "committed minority"
            _, _axes = joyplot(filtered_data,
                                    by="VoteCombination",
                                    column="Satisfaction",
                                    #   figsize=(10,5),
                                    overlap=overlap,
                                    ax=axes,
                                    colormap=ListedColormap(sns.color_palette('pastel', n_colors=3)),
                                    # colormap=plt.cm.YlGnBu,
                                    ylabels=use_ylabels,
                                    #   title=f'Vote scenario: {vote_scenario}, {scenario}',
                                    title=f'{vote_scenario}',
                                    medians=False,
                                    #   x_range=[all_sats['Satisfaction'].min(), all_sats['Satisfaction'].max()]
                                    x_range=[0,1]
                                    )

        h_pad = 5 + (- 5*(1 + overlap))
        fig.tight_layout(h_pad=h_pad)
        if group is not None:
            fig.savefig(f"{plots_directory}/{scenario}_ridge_{group}.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"{plots_directory}/{scenario}_ridge.pdf", bbox_inches="tight")
print(all_sats.groupby(['Scenario', 'VoteScenario','VoteCombination']).agg(['mean', 'std']).to_latex())
