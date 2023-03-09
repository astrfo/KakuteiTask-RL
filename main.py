import os
from datetime import datetime
from make_folder import compare_base_make_folder
from simulator import Simulator
from q_learning import QLearning
from q_learning_bias import QLearningBias


if __name__ == '__main__':
    algo = 'Qbias' #Q or Qbias
    simulation = 100
    episode = 10000
    alpha = 0.01
    beta = 5
    gamma = 0.9
    bias_weight = 5

    ex_param = {
        'sim': simulation,
        'epi': episode,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'biasWeight': bias_weight,
    }

    if algo == 'Q':
        policy = QLearning()
        results_dir_path = compare_base_make_folder(algo, ex_param)
    elif algo == 'Qbias':
        policy = QLearningBias()
        results_dir_path = compare_base_make_folder(algo, ex_param)
    else:
        print(f'Not found algorithm {algo}')
        exit(1)

    sim = Simulator(simulation, episode, results_dir_path)
    sim.run()