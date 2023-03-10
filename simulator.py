import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import Environment
from collector import Collector


class Simulator:
    def __init__(self, simulation, episode, policy, results_dir_path):
        self.sim = simulation
        self.epi = episode
        self.results_dir_path = results_dir_path
        self.env = Environment()
        self.collector = Collector(self.sim, self.epi, self.results_dir_path)
        self.algorithm = policy

    def run(self):
        for s in tqdm(range(self.sim)):
            self.collector.reset()
            self.algorithm.reset()
            for e in range(self.epi):
                state, done = self.env.reset()
                while not done:
                    action = self.algorithm.act(state)
                    next_state, reward, done = self.env.step(action)
                    Q, V, TD_Q, TD_V = self.algorithm.update(state, action, reward, next_state)
                    state = next_state
                    if action != 2: first_action = action
                self.collector.collect_episode_data(e, Q, V, TD_Q, TD_V, first_action)
            self.collector.save_episode_data()
        self.collector.save_simulation_data()