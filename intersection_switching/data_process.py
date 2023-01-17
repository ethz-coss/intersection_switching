import plotly.graph_objects as go
import plotly.offline as pyo
import pickle
import numpy as np

low_balanced = [11, 11]
low_unbalanced = [11, 6]

medium_balanced = [22, 22]
medium_unbalanced = [22, 11]

high_balanced = [32, 32]
high_unbalanced = [32, 16]


traffic_conditions = [low_balanced]#, low_unbalanced]#, medium_balanced, medium_unbalanced, high_balanced, high_unbalanced]


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

vote_types = [vote_stops]#, vote_stops, vote_wait]#, vote_uniform_1, vote_uniform_2, vote_uniform_3]#, vote_quarter_1, vote_quarter_2, vote_quarter_3, vote_quarter_4, vote_quarter_5, vote_quarter_6]

categories = ['Speed', 'Number of Stops', 'Wait Time']
categories = [*categories, categories[0]]

data = []

for traffic in traffic_conditions:
    for vote in vote_types:

        path = f"../runs/{traffic[0]}_{traffic[1]}_{vote[0]}_{vote[1]}_{vote[2]}"

        speeds_path = path + "/veh_speed_hist.pickle"
        stops_path = path + "/veh_stops.pickle"
        wait_path = path + "/veh_wait_time.pickle"

        with open(speeds_path, "rb") as f:
            speeds = pickle.load(f)
            
        avg_speed = np.mean([np.mean(speeds[x]) for x in speeds.keys()])
        var_speed = np.var(([np.mean(speeds[x]) for x in speeds.keys()]))
        print(avg_speed, var_speed)

        with open(stops_path, "rb") as f:
            stops = pickle.load(f)

        avg_stops = np.mean([np.mean(stops[x]) for x in stops.keys()])
        var_stops = np.var(([np.mean(stops[x]) for x in stops.keys()]))
        print(avg_stops, var_stops)

        with open(wait_path, "rb") as f:
            wait = pickle.load(f)

        print([wait[x] for x in wait.keys()])
        avg_wait = np.mean([np.mean(wait[x]) for x in wait.keys()])
        var_wait = np.var(([np.mean(wait[x]) for x in wait.keys()]))
        print(avg_wait, var_wait)


        result = [avg_speed, avg_stops, avg_wait]
        result = [*result, result[0]]
        data.append(go.Scatterpolar(r=result, theta=categories, name=f'{vote[0]}_{vote[1]}_{vote[2]}'))



    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=go.layout.Title(text='Restaurant comparison'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True
        )
    )

    pyo.plot(fig)
    fig.show()
