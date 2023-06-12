import os

# low_balanced = [11, 11]
# low_unbalanced = [11, 6]

# medium_balanced = [22, 22]
# medium_unbalanced = [22, 11]

# high_balanced = [32, 32]
# high_unbalanced = [32, 16]
# traffic_conditions = [medium_balanced]#, low_unbalanced, medium_balanced, medium_unbalanced, high_balanced, high_unbalanced]

# vote_speed = [1, 0, 0]
# vote_stops = [0, 1, 0]
# vote_wait = [0, 0, 1]
#
# vote_uniform_1 = [0.5, 0.5, 0]
# vote_uniform_2 = [0.5, 0, 0.5]
# vote_uniform_3 = [0, 0.5, 0.5]
#
# vote_quarter_1 = [0.75, 0.25, 0]
# vote_quarter_2 = [0.75, 0, 0.25]
# vote_quarter_3 = [0.25, 0, 0.75]
# vote_quarter_4 = [0, 0.25, 0.75]
# vote_quarter_5 = [0.25, 0.75, 0]
# vote_quarter_6 = [0, 0.75, 0.25]

scenarios = {
    'bipolar': [0, 0.5, 0.5],
    'majority_extreme': [0, 0.2, 0.8],
    'random': [0, 0.5, 0.5],
}

scenarios = {
    'debug_cumulative_majority': [0, 0.5, 0.5],
}

pure_methods = ['stops', 'waits']

input_methods = ['binary', 'cumulative']
configs = ['../scenarios/hangzhou/1.config', '../scenarios/ny16/1.config']
vote_types = ['proportional', 'majority']

total_points = 10
sim_steps = 3600
trials = 5

if __name__=='__main__':
    for sim_config in configs:
        for scenario, weights in scenarios.items():
            for input_method in input_methods:
                for vote_type in vote_types:
                    calls = []
                    for i in range(trials):
                        if input_method=='binary':
                        # if scenario in pure_methods:
                            vote_weights = " ".join(str(x) for x in weights)
                            call = f"python runner.py --sim_config {sim_config} --num_sim_steps {sim_steps} --seed {i} --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type learning --num_episodes 1 --vote_weights {vote_weights} --mfd False --scenario {scenario} --vote_type {vote_type} --path '../runs/{vote_type}/{input_method}/'"
                        else:  # cumulative voting
                            call = f"python runner.py --sim_config {sim_config} --num_sim_steps {sim_steps} --seed {i} --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type learning --num_episodes 1 --mfd False --total_points {total_points} --scenario {scenario} --vote_type {vote_type} --path '../runs/{vote_type}/{input_method}/'"
                        calls.append(call)

                    pycalls = "\n".join(calls)
                    os.system(f"""sbatch -n 8  --time=8:00:00 --wrap '{pycalls}'""")
