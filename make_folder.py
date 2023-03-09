import os
from datetime import datetime


def compare_base_make_folder(algo, ex_param):
    base_param = {
        'sim': 100,
        'epi': 1000,
        'alpha': 0.01,
        'beta': 5,
        'gamma': 0.9,
        'biasWeight': 1,
    }
    folder_name = algo
    for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
        if (base_k == ex_k) and (base_v != ex_v):
            folder_name += f'_{ex_k}{ex_v}'
    ex_folder_path = f'log/{folder_name}/'
    os.makedirs(ex_folder_path, exist_ok=True)
    time_now = datetime.now()
    results_dir_path = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
    os.makedirs(results_dir_path, exist_ok=True)
    return results_dir_path