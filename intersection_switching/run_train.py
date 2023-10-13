import os
from shutil import which

IS_SLURM = which('sbatch') is not None

# reward_types = ["stops", "speed", "wait"]
reward_types = ['stops', 'wait', 'localwait', "unique_stops"]
sim_steps = 3600
episodes = 400

configs = ['../scenarios/hangzhou/1.config',
           '../scenarios/hangzhou/2.config',
        #    '../scenarios/ny16/1.config'
          ]

learning_rates = {'stops': 0.005,'unique_stops': 0.005,
                  'wait': 0.025,
                  'localwait': 0.025}
decays = {'stops': 0.00005,
            'unique_stops': 0.00005,
                  'wait': 0.00005,
                  'localwait': 0.00005}
for config in configs:
    for reward in reward_types:
        # os.system("python runner.py --sim_config '../scenarios/loop_intersection/rings.config' --num_sim_steps 3600 --eps_start 1 --lr 0.0005 --mode train --agents_type learning --num_episodes 150 --replay True --mfd False --reward_type " + reward + " --n_vehs " + str(traffic[0]) + " " + str(traffic[1]))

# 'python runner.py --sim_config ../scenarios/2x2/1.config --num_sim_steps 3600 --eps_start 1 --lr 0.0005 --mode train --agents_type learning --num_episodes 100 --replay True --mfd False --reward_type stops'

        pycalls = f"python runner.py --sim_config {config} --num_sim_steps {sim_steps} --eps_start 1 --eps_decay {decays[reward]} --lr {learning_rates[reward]} --mode train --agents_type learning --num_episodes {episodes} --mfd False --reward_type {reward}"
        if IS_SLURM:
            os.system(f"sbatch -n 4  --mem-per-cpu=4G --time=9:00:00 --wrap '{pycalls}'")
        else:
            os.system(f'{pycalls} &')
    #     break
    # break