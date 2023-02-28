import os
from datetime import datetime
from simulator import Simulator

def make_folder():
    log_path = f'log/'
    os.makedirs(log_path, exist_ok=True)
    time_now = datetime.now()
    results_dir_path = f'{log_path}{time_now:%Y%m%d%H%M}/'
    os.makedirs(results_dir_path, exist_ok=True)
    return results_dir_path

def main():
    results_dir_path = make_folder()
    simulation = 100
    episode = 10000
    sim = Simulator(simulation, episode, results_dir_path)
    sim.run()

if __name__ == '__main__':
    print('started run')
    main()
    print('finished run')