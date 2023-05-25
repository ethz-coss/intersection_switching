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
    'balanced_mild': [0, 0.5, 0.5],
    'majority_mild': [0, 0.6, 0.4],
    'majority_extreme': [0, 0.2, 0.8],
}

input_methods = ['binary', 'cumulative']
configs = ['../scenarios/hangzhou/1.config', '../scenarios/ny16/1.config']
total_points = 10

for sim_config in configs:
    for scenario, weights in scenarios.items():
        for input_method in input_methods:
            calls = []
            for i in range(100):
                if input_method == 'binary':
                    vote_weights = " ".join(str(x) for x in weights)
                    call = f"python runner.py --sim_config {sim_config} --num_sim_steps 3600 --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type learning --num_episodes 1 --vote_weights {vote_weights} --replay False --mfd False --path '../runs/proportional/'"
                else:  # cumulative voting
                    call = f"python runner.py --sim_config {sim_config} --num_sim_steps 3600 --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type learning --num_episodes 1  --replay False --mfd False --total_points {total_points} --scenario {scenario} --path '../runs/proportional/'"
                calls.append(call)

            pycalls = "\n".join(calls)
            os.system(f"""sbatch -n 8 --wrap '{pycalls}'""")
