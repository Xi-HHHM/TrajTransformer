import numpy as np

from utils.util_mp import *
import matplotlib.pyplot as plt
from utils.util_data import *
from utils.util_wandb import plot_reconstruction
import h5py
import time
import os


def load_data_as_array(path=None):
    # FixMe
    # Get python package path
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, 'data')

    # Get all hdf5 files in the data directory
    files = [f for f in os.listdir(path) if f.endswith('.h5') and "random_problems_w301" in f]

    list_traj_301 = []
    list_traj_51 = []
    list_waypoints = []
    for file in files:
        with h5py.File(os.path.join(path, file), 'r') as f:
            print(f.keys())
            traj_301 = f['traj_301'][:]
            traj_51 = f['traj_51'][:]
            waypoints = f['waypoint'][:]

        list_traj_301.append(traj_301)
        list_traj_51.append(traj_51)
        list_waypoints.append(waypoints)

    return list_traj_301, list_traj_51, list_waypoints


def get_waypoints_indices(waypoints, trajectories):
    indices = []
    starting_index = 0
    n_traj = trajectories.shape[0]
    n_wp = waypoints.shape[0]

    for i in range(n_traj):
        goal = trajectories[i, -1, :]
        for j in range(starting_index+1, n_wp):
            wp = waypoints[j, :]
            if np.all(wp-goal < 1e-6):
                indices.append(np.array([starting_index, j]))
                starting_index = j + 1
                break
    return indices


def combine_and_save_data(save_name, dir_name=None):
    # FixMe
    # Get python package path
    if dir_name is None:
        path = os.path.dirname(os.path.abspath(__file__))
        dir_name = os.path.join(path, 'data')

    list_traj_301, list_traj_51, list_waypoints = load_data_as_array(dir_name)
    # Save the data
    np_traj_301 = np.vstack(list_traj_301)
    np_traj_51 = np.vstack(list_traj_51)

    if np_traj_51.ndim == 2:
        np_traj_51 = np.reshape(np_traj_51, [-1, 51, 6])
    if np_traj_301.ndim == 2:
        np_traj_301 = np.reshape(np_traj_301, [-1, 301, 6])

    np_waypoints = np.vstack(list_waypoints)

    list_indices = get_waypoints_indices(np_waypoints, np_traj_51)
    np_list_indices = np.vstack(list_indices)


    with h5py.File(os.path.join(dir_name, "301_" + save_name), 'w') as f:
        f.create_dataset('traj', data=np_traj_301, maxshape=(None, np_traj_301.shape[1], 6), chunks=True,
                         compression='gzip')
        f.create_dataset('waypoints', data=np_waypoints, maxshape=(None, 6), chunks=True,
                            compression='gzip')
        f.create_dataset('indices', data=np_list_indices, maxshape=(None, 2), chunks=True,
                            compression='gzip')

    with h5py.File(os.path.join(dir_name, "51_" + save_name), 'w') as f:
        f.create_dataset('traj', data=np_traj_51, maxshape=(None, np_traj_51.shape[1], 6), chunks=True,
                         compression='gzip')
        f.create_dataset('waypoints', data=np_waypoints, maxshape=(None, 6), chunks=True,
                            compression='gzip')
        f.create_dataset('indices', data=np_list_indices, maxshape=(None, 2), chunks=True,
                            compression='gzip')


if __name__ == "__main__":
    combine_and_save_data(save_name="random_problems_waypoint_and_parameterized.h5")