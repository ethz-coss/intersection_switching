import os

# low_balanced = [11, 11]
# low_unbalanced = [11, 6]

# medium_balanced = [22, 22]
# medium_unbalanced = [22, 11]

# high_balanced = [32, 32]
# high_unbalanced = [32, 16]


# traffic_conditions = [medium_balanced]#, low_unbalanced, medium_balanced, medium_unbalanced, high_balanced, high_unbalanced]


vote_speed = [1, 0, 0]
vote_stops = [0, 1, 0]
vote_wait = [0, 0, 1]

vote_uniform_1 = [0.5, 0.5, 0]
vote_uniform_2 = [0.5, 0, 0.5]
vote_uniform_3 = [0, 0.5, 0.5]

vote_quarter_1 = [0.75, 0.25, 0]
vote_quarter_2 = [0.75, 0, 0.25]
vote_quarter_3 = [0.25, 0, 0.75]
vote_quarter_4 = [0, 0.25, 0.75]
vote_quarter_5 = [0.25, 0.75, 0]
vote_quarter_6 = [0, 0.75, 0.25]


# vote_types = [vote_stops, vote_wait, vote_uniform_1, vote_uniform_2, vote_uniform_3]#, vote_quarter_1, vote_quarter_2, pvote_quarter_3, vote_quarter_4, vote_quarter_5, vote_quarter_6]
# vote_types = [vote_speed]
# vote_types = [vote_uniform_3]
vote_types = [vote_stops, vote_wait, vote_uniform_3]

configs = ['../scenarios/hangzhou/1.config',
           '../scenarios/ny16/1.config']
for sim_config in configs:
    for vote in vote_types:
        vote = [str(i) for i in vote]
        # for traffic in traffic_conditions:
        calls = []
        for i in range(1):
            call = f"python runner.py --sim_config {sim_config} --num_sim_steps 3600 --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type learning --num_episodes 1 --replay False --mfd False  --vote_weights {' '.join(vote)} --vote_type proportional --path '../runs/proportional/'"
            calls.append(call)

        pycalls = "\n".join(calls)
        os.system(f"""sbatch -n 8 --wrap '{pycalls}'""")

# python runner.py --sim_config ../scenarios/2x2/1.config --num_sim_steps 3600 --eps_start 1 --lr 0.0005 --mode train --agents_type learning --num_episodes 100 --replay True --mfd False --reward_type wait
        # os.system("sbatch -n 8 --wrap \"python runner.py --sim_config '../scenarios/loop_intersection/rings.config' --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode vote --agents_type learning --num_episodes 1 --replay True --mfd False " + " --n_vehs " + str(traffic[0]) + " " + str(traffic[1]) + " --vote_weights " + vote_weights[0] + " " + vote_weights[1] + " " + vote_weights[2] + " " + "\"" )
