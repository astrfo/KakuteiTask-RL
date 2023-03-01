import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import Environment
from q_learning import QLearning
from q_learning_bias import QLearningBias

class Simulator:
    def __init__(self, simulation, episode, results_dir_path):
        self.sim = simulation
        self.epi = episode
        self.results_dir_path = results_dir_path
        self.env = Environment()
        self.algorithm = QLearning()
        # self.algorithm = QLearningBias()
        self.Q = np.zeros((self.sim, self.epi, 5)) #Q_HL, Q_HR, Q_LL, Q_LN, Q_LR
        self.V = np.zeros((self.sim, self.epi, 4)) #START, LL, LN, LR
        self.td_error_q = np.zeros((self.sim, self.epi, 5)) #Q_HL, Q_HR, Q_LL, Q_LN, Q_LR
        self.td_error_v = np.zeros((self.sim, self.epi, 4)) #START, LL, LN, LR
        self.action_count = np.zeros((self.sim, self.epi, 2)) #HL, HR

    def run(self):
        for s in tqdm(range(self.sim)):
            self.algorithm.reset()
            for e in range(self.epi):
                state, done = self.env.reset() #state=START, #doneは終端状態=True, Not終端状態=False
                while not done:
                    action = self.algorithm.act(state)
                    next_state, reward, done = self.env.step(action)
                    Q, V, td_error_q, td_error_v = self.algorithm.update(state, action, reward, next_state)
                    self.Q[s, e] = Q
                    self.V[s, e] = V
                    self.td_error_q[s, e] = td_error_q
                    self.td_error_v[s, e] = td_error_v
                    state = next_state
                    if action != 2: self.action_count[s, e, action] = 1
        self.plot_Q()
        self.plot_V()
        self.plot_td_error_q()
        self.plot_td_error_v()
        self.plot_rate()

    def plot_Q(self):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(np.arange(self.epi), np.mean(self.Q[:, :, 0], axis=0), label='Q_HL')
        plt.plot(np.arange(self.epi), np.mean(self.Q[:, :, 1], axis=0), label='Q_HR')
        plt.plot(np.arange(self.epi), np.mean(self.Q[:, :, 2], axis=0), label='Q_LL_TN')
        plt.plot(np.arange(self.epi), np.mean(self.Q[:, :, 3], axis=0), label='Q_LN_TN')
        plt.plot(np.arange(self.epi), np.mean(self.Q[:, :, 4], axis=0), label='Q_LR_TN')
        plt.title(f'{self.algorithm.__class__.__name__}, sim={self.sim}, epi={self.epi}')
        plt.xlabel('episode')
        plt.ylabel('average Q')
        # plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig(f'{self.results_dir_path}Q.png')
        plt.close()

    def plot_V(self):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(np.arange(self.epi), np.mean(self.V[:, :, 0], axis=0), label='V_START')
        plt.plot(np.arange(self.epi), np.mean(self.V[:, :, 1], axis=0), label='V_LL')
        plt.plot(np.arange(self.epi), np.mean(self.V[:, :, 2], axis=0), label='V_LN')
        plt.plot(np.arange(self.epi), np.mean(self.V[:, :, 3], axis=0), label='V_LR')
        plt.title(f'{self.algorithm.__class__.__name__}, sim={self.sim}, epi={self.epi}')
        plt.xlabel('episode')
        plt.ylabel('average V')
        # plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig(f'{self.results_dir_path}V.png')
        plt.close()

    def plot_td_error_q(self):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(np.arange(self.epi), np.mean(self.td_error_q[:, :, 0], axis=0), label='TD_Q_HL', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_q[:, :, 1], axis=0), label='TD_Q_HR', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_q[:, :, 2], axis=0), label='TD_Q_LL_TN', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_q[:, :, 3], axis=0), label='TD_Q_LN_TN', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_q[:, :, 4], axis=0), label='TD_Q_LR_TN', alpha=0.5)
        plt.title(f'{self.algorithm.__class__.__name__}, sim={self.sim}, epi={self.epi}')
        plt.xlabel('episode')
        plt.ylabel('average TD_Q')
        # plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig(f'{self.results_dir_path}TD_Q.png')
        plt.close()

    def plot_td_error_v(self):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(np.arange(self.epi), np.mean(self.td_error_v[:, :, 0], axis=0), label='TD_V_START', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_v[:, :, 1], axis=0), label='TD_V_LL', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_v[:, :, 2], axis=0), label='TD_V_LN', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.td_error_v[:, :, 3], axis=0), label='TD_V_LR', alpha=0.5)
        plt.title(f'{self.algorithm.__class__.__name__}, sim={self.sim}, epi={self.epi}')
        plt.xlabel('episode')
        plt.ylabel('average TD_V')
        # plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig(f'{self.results_dir_path}TD_V.png')
        plt.close()

    def plot_rate(self):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(np.arange(self.epi), np.mean(self.action_count[:, :, 0], axis=0), label='HL', alpha=0.5)
        plt.plot(np.arange(self.epi), np.mean(self.action_count[:, :, 1], axis=0), label='HR', alpha=0.5)
        plt.title(f'{self.algorithm.__class__.__name__}, sim={self.sim}, epi={self.epi}')
        plt.xlabel('episode')
        plt.ylabel('average action count')
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.savefig(f'{self.results_dir_path}action_rate.png')
        plt.close()
