import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import Environment
from q_learning import QLearning

class Simulator:
    def __init__(self, simulation, episode):
        self.sim = simulation
        self.epi = episode
        self.env = Environment()
        self.algorithm = QLearning()
        self.Q_values = np.zeros((self.sim, self.epi, 2)) #Q_HL, Q_HR

    def run(self):
        for s in tqdm(range(self.sim)):
            self.algorithm.reset()
            for e in range(self.epi):
                state, done = self.env.reset() #state=START=, #doneは終端状態=True, Not終端状態=False
                while not done:
                    action = self.algorithm.act(state)
                    reward, next_state = self.env.step(state, action)
                    Q_HL, Q_HR, done = self.algorithm.update(state, action, reward, next_state)
                    self.Q_values[s, e] = Q_HL, Q_HR
                    state = next_state
                    if done:
                        break
        self.plot_Q_values()

    def plot_Q_values(self):
        plt.plot(np.arange(self.epi), np.mean(self.Q_values[:, :, 0], axis=0), label='Q_HL')
        plt.plot(np.arange(self.epi), np.mean(self.Q_values[:, :, 1], axis=0), label='Q_HR')
        plt.title(f'Q-learning, sim={self.sim}, epi={self.epi}')
        plt.xlabel('episode')
        plt.ylabel('average Q value')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
