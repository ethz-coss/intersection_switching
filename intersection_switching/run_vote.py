import os
from shutil import which

IS_SLURM = which('sbatch') is not None


vote_scenarios = {
    'bipolar': [0, 0.5, 0.5],
    'majority_extreme': [0, 0.2, 0.8],
    'random': [0, 0.5, 0.5],
    # 'stops': [0, 1, 0],
    'unique_stops': [0, 1, 0],
    # 'waits': [0, 0, 1], 
    'localwait': [0, 0, 1],
    # 'demand': [0, 0.5, 0.5],
    # 'fixed': [0, 0.5, 0.5],
}


pure_methods = [
                # 'stops', 
                # 'waits', 
                'localwait', 
                'unique_stops'
                ]

# baseline = ['demand', 'fixed']
baseline = ['fixed']
input_methods = ['binary', 'cumulative']
# configs = ['../scenarios/hangzhou/1.config', '../scenarios/ny16/1.config']
configs = ['../scenarios/hangzhou/1.config',
            '../scenarios/hangzhou/2.config']
vote_types = ['proportional', 'majority']

total_points = 10
sim_steps = 3600
trials = 10

if __name__=='__main__':
    for sim_config in configs:
        override = False
        for scenario, weights in vote_scenarios.items():
            for input_method in input_methods:
                for vote_type in vote_types:
                    calls = []
                    agent_type = 'learning'
                    breakflag = False # do only one of pure and baselines
                    for i in range(trials):
                        path = f'../runs/{vote_type}/{input_method}{"_override" if override else ""}/'
                        if scenario in pure_methods:
                            path = f'../runs/pure/{scenario}'
                            # breakflag = True
                        if scenario in baseline:
                            path = f'../runs/baseline/{scenario}'
                            # breakflag = True
                            agent_type = scenario
                        if input_method=='binary':
                            vote_weights = " ".join(str(x) for x in weights)
                            call = f"python runner.py --sim_config {sim_config} --num_sim_steps {sim_steps} --seed {i} --ID {i} --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type {agent_type} --num_episodes 1 --mfd False --total_points {total_points} --binary --scenario {scenario} --vote_type {vote_type} --path {path}"
                        else:  # cumulative voting
                            call = f"python runner.py --sim_config {sim_config} --num_sim_steps {sim_steps} --seed {i} --ID {i} --eps_start 0 --eps_end 0 --lr 0.0005 --mode vote --agents_type {agent_type} --num_episodes 1 --mfd False --total_points {total_points} --scenario {scenario} --vote_type {vote_type} --path {path}"
                        
                        if override:
                            call += ' --override'
                        if i==0: # replay
                            call += ' --trajectory --replay'
                        calls.append(call)
                        if breakflag:
                            break

                    if IS_SLURM:
                        pycalls = "\n".join(calls)
                        # print(pycalls)
                        os.system(f"""sbatch -n 4  --mem-per-cpu=2G --time=4:00:00 --wrap '{pycalls}'""")
                    else:
                        pycalls = "&".join(calls)
                        os.system(pycalls)
                    if breakflag:
                        break
                if breakflag:
                    break
