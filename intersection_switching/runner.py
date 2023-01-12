import numpy as np
import random
import argparse

from models.dqn import DQN
from environ import Environment
from logger import Logger
from importlib import import_module
import torch

SEED = 2


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        raise ImportError(msg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_config", default='../scenarios/2x2/1.config',
                        type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=1, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)"
                        )
    parser.add_argument("--num_sim_steps", default=1800, type=int,
                        help="the number of simulation steps, one step corresponds to 1 second")
    parser.add_argument("--agents_type", default='analytical', type=str,
                        help="the type of agents learning/policy/analytical/hybrid/demand")
    parser.add_argument("--rl_model", default='dqn', type=str,
                        help="rl algorithm used, defaults to deep Q-learning")
    parser.add_argument("--update_freq", default=10, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=10")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=float,
                        help="the learning rate for the dqn, default=5e-4")
    parser.add_argument("--eps_start", default=1,
                        type=float, help="the epsilon start")
    parser.add_argument("--eps_end", default=0.01,
                        type=float, help="the epsilon decay")
    parser.add_argument("--eps_decay", default=5e-5,
                        type=float, help="the epsilon decay")
    parser.add_argument("--eps_update", default=1799,
                        type=float, help="how frequently epsilon is decayed")
    parser.add_argument("--load", default=None, type=str,
                        help="path to the model to be loaded")
    parser.add_argument("--mode", default='train', type=str,
                        help="mode of the run train/test")
    parser.add_argument("--replay", default=False,
                        type=bool, help="saving replay")
    parser.add_argument("--mfd", default=False, type=bool,
                        help="saving mfd data")
    parser.add_argument("--path", default='../runs/', type=str,
                        help="path to save data")
    parser.add_argument("--meta", default=False, type=bool,
                        help="indicates if meta learning for ML")
    parser.add_argument("--load_cluster", default=None, type=str,
                        help="path to the clusters and models to be loaded")
    parser.add_argument("--ID", default=None, type=int,
                        help="id used for naming")
    parser.add_argument("--gamma", default=0.8, type=float,
                        help="gamma parameter for the DQN")
    parser.add_argument("--reward_type", default='speed', type=str,
                        help="reward function for the agent")
    return parser.parse_args()


def run_exp(environ, args, num_episodes, num_sim_steps, policy, logger, detailed_log=False):
    step = 0
    best_time = 999999
    best_veh_count = 0
    best_reward = -999999
    saved_model = None
    environ.best_epoch = 0

    environ.eng.set_save_replay(open=False)
    environ.eng.set_random_seed(2)
    random.seed(2)
    np.random.seed(2)

    log_phases = False

    print(f'actions: {list(environ.intersections.values())[0].action_space.n}')
    for i_episode in range(num_episodes):
        logger.losses = []
        if i_episode == num_episodes-1 and args.replay:
            environ.eng.set_save_replay(open=True)
            environ.eng.set_replay_file(f"../{logger.log_path}_{args.reward_type}/replay_file.txt")

        print("episode ", i_episode)

        obs = environ.reset()

        while environ.time < num_sim_steps:
            # Dispatch the observations to the model to get the tuple of actions
            # actions = {id: 1*(np.random.random()>0.5) for id in environ.agent_ids} # random policy

            if environ.agents_type in ['learning']:
                actions = {}
                for agent_id in environ.agent_ids:
                    act = policy.act(torch.FloatTensor(
                        obs[agent_id], device=device), epsilon=environ.eps)
                actions[agent_id] = act


            # Execute the actions
            next_obs, rewards, dones, info = environ.step(actions)
            # print(next_obs, rewards)

            # Update the model with the transitions observed by each agent
            
            if args.mode=='train' and environ.agents_type in ['learning']:
                step = (step+1) % environ.update_freq
                if environ.time > 50:
                    for agent_id in rewards.keys():
                        state = torch.FloatTensor(obs[agent_id], device=device)
                        reward = torch.tensor(
                            [rewards[agent_id]], dtype=torch.float, device=device)
                        done = torch.tensor(
                            [dones[agent_id]], dtype=torch.bool, device=device)
                        action = torch.tensor(
                            [actions[agent_id]], device=device)
                        next_state = torch.FloatTensor(
                            next_obs[agent_id], device=device)
                        policy.memory.add(
                            state, action, reward, next_state, done)

                if step == 0:
                    tau = 1e-3
                    _loss = 0
                    _loss -= policy.optimize_model(
                        gamma=args.gamma, tau=tau)
                    logger.losses.append(-_loss)
                    environ.eps = max(environ.eps-environ.eps_decay, environ.eps_end)
            obs = next_obs

            environ.agent_history.append(act)

        if environ.agents_type in ['learning']:
            if environ.eng.get_average_travel_time() < best_time:
                best_time = environ.eng.get_average_travel_time()
                logger.save_models([policy], flag=False)
                environ.best_epoch = i_episode

            if environ.eng.get_finished_vehicle_count() > best_veh_count:
                best_veh_count = environ.eng.get_finished_vehicle_count()
                logger.save_models([policy], flag=True)
                environ.best_epoch = i_episode

            # if logger.reward > best_reward:
            best_reward = logger.reward
            logger.save_models([policy], flag=None)
        logger.log_measures(environ)
        logger.log_delays(args.sim_config, environ)

        print_string = (f'Rew: {logger.reward:.4f}\t'
                        f'Vehicles: {len(environ.vehicles):.0f}\t'
                        f'Speed (m/s): {np.mean(environ.speeds):.2f}\t'
                        f'Stops (total): {np.sum(environ.stops):.2f}\t'
                        f'WaitTimes (sec): {np.mean(environ.waiting_times):.2f}\t'
                        )
        if True:
            print_string += f'Delay (sec/km): {np.mean(logger.delays[-1]):.2f}'
        print(print_string)

    # logger.save_log_file(environ)
    logger.serialise_data(environ, policies[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = parse_args()
    logger = Logger(args)

    environ = Environment(args, args.reward_type)

    act_space = environ.action_space
    obs_space = environ.observation_space
    print(act_space.n, obs_space.shape)
    if args.agents_type in ['learning']:
        policy = DQN(obs_space, act_space, seed=SEED, load=args.load)
    else:
        print('not using a policy')
        policy = None
    policies = [policy]

    num_episodes = args.num_episodes
    num_sim_steps = args.num_sim_steps


    detailed_log = args.mode == 'test'
    run_exp(environ, args, num_episodes, num_sim_steps, policies[0], logger, detailed_log)

