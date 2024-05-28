import numpy as np
import h5py


def load_traj_from_hdf5(hdf5_file, sort=True):
    with h5py.File(hdf5_file, 'r') as f:
        wp_keys = list(f.keys())
        if sort:
            wp_keys.sort(key=lambda x: int(x))
        traj_list = [np.array(f[key]) for key in wp_keys]
    return traj_list


def load_data(path=None):
    # FixMe
    # Get python package path
    if path is None:
        import os
        path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path, '../data')

    # Get all hdf5 files in the data directory
    files = [f for f in os.listdir(data_path) if f.endswith('.h5') and "random_problems" in f]
    data = []
    file = files[0]
    with h5py.File(os.path.join(data_path, file), 'r') as f:
        traj = f['traj'][:]

    data = np.split(traj, traj.shape[0])
    n_data = len(data)
    train_data = data[:int(0.8 * n_data)]
    test_data = data[int(0.8 * n_data):]

    return train_data, test_data


def load_data_as_array(path=None):
    # FixMe
    # Get python package path
    if path is None:
        import os
        path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path, '../data')

    # Get all hdf5 files in the data directory
    files = [f for f in os.listdir(data_path) if f.endswith('.h5') and "random_problems" in f]
    data = []
    file = files[0]
    with h5py.File(os.path.join(data_path, file), 'r') as f:
        traj = f['traj'][:]

    return traj