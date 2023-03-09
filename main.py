import os
from datetime import datetime
from make_folder import compare_base_make_folder
from simulator import Simulator
from policy import QLearning, QLearningTDbias


if __name__ == '__main__':
    algo = 'Q_TD' #Q or Q_TD
    simulation = 100
    episode = 10000
    alpha = 0.01
    beta = 5
    gamma = 0.9
    bias_weight = 1

    ex_param = {
        'sim': simulation,
        'epi': episode,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'biasWeight': bias_weight,
    }

    if algo == 'Q':
        policy = QLearning(**ex_param)
        results_dir_path = compare_base_make_folder(algo, ex_param)
    elif algo == 'Q_TD':
        policy = QLearningTDbias(**ex_param)
        results_dir_path = compare_base_make_folder(algo, ex_param)
    else:
        print(f'Not found algorithm {algo}')
        exit(1)

    sim = Simulator(simulation, episode, policy, results_dir_path)
    sim.run()