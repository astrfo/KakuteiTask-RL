import numpy as np
import matplotlib.pyplot as plt


class Collector:
    def __init__(self, sim, epi, results_dir_path, is_save_img=True):
        self.sim = sim
        self.epi = epi
        self.results_dir_path = results_dir_path
        self.is_save_img = is_save_img

        self.Q_sim = np.zeros((self.epi, 5))
        self.V_sim = np.zeros((self.epi, 4))
        self.TD_Q_sim = np.zeros((self.epi, 5))
        self.TD_V_sim = np.zeros((self.epi, 4))
        self.action_rate_sim = np.zeros((self.epi, 2))

    def reset(self):
        self.Q_epi = np.zeros((self.epi, 5))
        self.V_epi = np.zeros((self.epi, 4))
        self.TD_Q_epi = np.zeros((self.epi, 5))
        self.TD_V_epi = np.zeros((self.epi, 4))
        self.action_rate_epi = np.zeros((self.epi, 2))

    def collect_episode_data(self, e, Q, V, TD_Q, TD_V, action):
        self.Q_epi[e] += Q
        self.V_epi[e] += V
        self.TD_Q_epi[e] += TD_Q
        self.TD_V_epi[e] += TD_V
        self.action_rate_epi[e, action] += 1

    def save_episode_data(self):
        self.Q_sim += self.Q_epi
        self.V_sim += self.V_epi
        self.TD_Q_sim += self.TD_Q_epi
        self.TD_V_sim += self.TD_V_epi
        self.action_rate_sim += self.action_rate_epi

    def collect_simulation_data(self):
        pass

    def save_simulation_data(self):
        self.Q_sim /= self.sim
        self.V_sim /= self.sim
        self.TD_Q_sim /= self.sim
        self.TD_V_sim /= self.sim
        self.action_rate_sim /= self.sim
        self.save_csv_img()

    def save_csv_img(self):
        np.savetxt(f'{self.results_dir_path}Q.csv', self.Q_sim, delimiter=',')
        np.savetxt(f'{self.results_dir_path}V.csv', self.V_sim, delimiter=',')
        np.savetxt(f'{self.results_dir_path}TD_Q.csv', self.TD_Q_sim, delimiter=',')
        np.savetxt(f'{self.results_dir_path}TD_V.csv', self.TD_V_sim, delimiter=',')
        np.savetxt(f'{self.results_dir_path}action_rate.csv', self.action_rate_sim, delimiter=',')
        if self.is_save_img: self.save_img()

    def save_img(self):
        self.sub_plot('Q', self.Q_sim)
        self.sub_plot('V', self.V_sim)
        self.sub_plot('TD_Q', self.TD_Q_sim)
        self.sub_plot('TD_V', self.TD_V_sim)
        self.sub_plot('action_rate', self.action_rate_sim)

    def sub_plot(self, title, thing):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(thing, alpha=0.6)
        plt.title(title)
        plt.xlabel('episode')
        plt.ylim(-0.1, 1.1)
        plt.savefig(f'{self.results_dir_path}{title}.png')
        plt.close()