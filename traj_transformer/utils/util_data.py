import numpy as np
import h5py


def load_traj_from_hdf5(hdf5_file, sort=True):
    with h5py.File(hdf5_file, 'r') as f:
        wp_keys = list(f.keys())
        if sort:
            wp_keys.sort(key=lambda x: int(x))
        traj_list = [np.array(f[key]) for key in wp_keys]
    return traj_list