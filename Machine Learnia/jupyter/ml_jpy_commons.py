# useful imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('dark_background')
import numpy as np
import seaborn as sns
# ...


# memory assets
from sys import getsizeof
KiB = 2**10
MiB = KiB * KiB
GiB = MiB * KiB
TiB = GiB * KiB
def format_iB(n_bytes):
    if n_bytes < KiB:
        return n_bytes, 'iB'
    elif n_bytes < MiB:
        return round(n_bytes / KiB, 3), 'KiB'
    elif n_bytes < GiB:
        return round(n_bytes / MiB, 3), 'MiB'
    elif n_bytes < TiB:
        return round(n_bytes / GiB, 3), 'GiB'
    else:
        return round(n_bytes / TiB), 'TiB'


# projects subdirs management
def print_path_info(path):
    print(path, 'exists' if os.path.exists(path) else 'doesn\'t exist')

def create_subdir(project_path, rel_path=''):
    path = os.path.join(project_path, rel_path)
    print_path_info(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'created.')
    return path

# project data dirs
project_dir = os.path.abspath('..')

data_dir = create_subdir(os.path.join(project_dir, 'data'))
csv_data_dir = create_subdir(os.path.join(data_dir, 'csv'))
