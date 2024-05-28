from utils.util_mp import *
import matplotlib.pyplot as plt
from utils.util_data import *
from utils.util_wandb import plot_reconstruction
import h5py
import time
import os


def test_reconstruct():
    mp = MP4Transformer()
    _, test_data = load_data()
    # traj, params, init_pos, init_vel = load_reconstruct()
    for i, d in enumerate(test_data):
        w = mp.get_mp_weights(d)
        recon_data = mp.get_prodmp_results(w['params'], w['init_pos'], w['init_vel'])
        # print(params[i] - w['params'].numpy())
        # print(init_pos[i] - w['init_pos'].numpy())
        # print(init_vel[i] - w['init_vel'].numpy())

        plot_reconstruction(d, recon_data["pos"], epoch=0, show=True)
        # plot_reconstruction(d, traj[i], epoch=0, show=True)


def save_reconstruct():
    mp = MP4Transformer()
    train_data, test_data = load_data()
    data = train_data + test_data
    with h5py.File('reconstructions_weight_conditions.h5', 'w') as f:
        for i, d in enumerate(data):
            w = mp.get_mp_weights(d)
            f.create_dataset(f'{i}_params', data=w['params'])
            f.create_dataset(f'{i}_init_pos', data=w['init_pos'])
            f.create_dataset(f'{i}_init_vel', data=w['init_vel'])


def save_reconstruct_2():
    mp = MP4Transformer()
    data = load_data_as_array()

    w = mp.get_mp_weights(data, reg=1e-5)
    params = w['params']
    init_pos = w['init_pos']
    init_vel = w['init_vel']

    reconstructed = mp.get_prodmp_results(params, init_pos, init_vel)
    traj = reconstructed['pos']

    pkg_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(pkg_path, 'data/reconstructions_weight_conditions_traj.h5')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('params', data=params, shape=params.shape, chunks=True,
                         compression='gzip', compression_opts=9)
        f.create_dataset('init_pos', data=init_pos, shape=init_pos.shape, chunks=True,
                         compression='gzip', compression_opts=9)
        f.create_dataset('init_vel', data=init_vel, shape=init_vel.shape, chunks=True,
                         compression='gzip', compression_opts=9)
        f.create_dataset('traj', data=data, shape=data.shape, chunks=True,
                         compression='gzip', compression_opts=9)
        # f.create_dataset('reconstructed', data=traj, shape=traj.shape, chunks=True,
        #                  compression='gzip', compression_opts=9)

def save_reconstruct_3():
    mp = MP4Transformer()
    train_data, test_data = load_data()
    data = train_data + test_data
    params = []
    init_pos = []
    init_vel = []
    for i, d in enumerate(data):
        w = mp.get_mp_weights(d)
        params.append(w['params'])
        init_pos.append(w['init_pos'])
        init_vel.append(w['init_vel'])

    # Convert to numpy arrays
    params = np.array(params)
    init_pos = np.array(init_pos)
    init_vel = np.array(init_vel)
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(pkg_path, 'data/reconstructions_weight_conditions_2.h5')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('params', data=params, maxshape=(None, params.shape[1]), chunks=True)
        f.create_dataset('init_pos', data=init_pos, maxshape=(None, init_pos.shape[1]), chunks=True)
        f.create_dataset('init_vel', data=init_vel, maxshape=(None, init_vel.shape[1]), chunks=True)


def load_reconstruct():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(pkg_path, 'data/reconstructions_weight_conditions_traj_220k_04W.h5')
    start_time = time.time()
    with h5py.File(file_path, 'r') as f:
        n_data = len(f.keys()) // 3
        for i in range(n_data):
            params = f[f'params'][:]
            init_pos = f[f'init_pos'][:]
            init_vel = f[f'init_vel'][:]
            traj = f[f'traj'][:]

    print(init_vel.shape)
    print(traj.shape)

    # Split the data and reduce the dimension
    n_data = init_vel.shape[0]
    params = np.split(params, n_data)
    init_pos = np.split(init_pos, n_data)
    init_vel = np.split(init_vel, n_data)

    print(n_data)
    print(len(params))
    print(params[0].shape)

    mp = MP4Transformer()
    index = 211531
    w = mp.get_mp_weights(traj[index], reg=1e-9)
    # reconstructed = mp.get_prodmp_results(params[index], init_pos[index], init_vel[index])
    reconstructed = mp.get_prodmp_results(w["params"], w["init_pos"], w["init_vel"])
    plot_reconstruction(traj[index], reconstructed["pos"], epoch=0, show=True)

    print(f"Time taken to load weight files: {time.time() - start_time:.2f}s")
    return traj, params, init_pos, init_vel


if __name__ == "__main__":
    load_reconstruct()