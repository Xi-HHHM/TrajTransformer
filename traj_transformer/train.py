import torch
import os

import wandb

from model import TrajReconstructor
from utils.util_mp import MP4Transformer
from utils.util_data import load_traj_from_hdf5
from utils.util_wandb import get_wandb_logger
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, list):
            data_dict = {i: d for i, d in enumerate(data)}

            # Reverse the numpy arrays in data
            # TODO: we need to check this
            data_dict_2 = {i + len(data): d[::-1, :].copy() for i, d in enumerate(data)}

            # Concatenate the two dictionaries
            data_dict.update(data_dict_2)

            self.data = data_dict
        else:
            raise ValueError("Data should be a list")

    def __len__(self):
        # Return the total number of data samples
        return len(self.data)

    def __getitem__(self, index):
        # Generate one sample of data
        return self.data[index]


def load_data():
    # FixMe
    # Get python package path
    path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(path, 'data')

    # Get all hdf5 files in the data directory
    files = [f for f in os.listdir(data_path) if f.endswith('.h5')]
    data = []
    for file in files:
        file = os.path.join(data_path, file)
        data_tmp = load_traj_from_hdf5(file)  # returns a list of numpy arrays
        data.extend(data_tmp)
    n_data = len(data)
    train_data = data[:int(0.8 * n_data)]
    test_data = data[int(0.8 * n_data):]

    return train_data, test_data


def dim_policy_out():
    """
    Get the dimension of the policy output

    Args:
        cfg: config dict

    Returns:
        dim_out: dimension of the policy output

    """
    dof = 6
    num_basis = 3
    learn_tau = False
    learn_delay = False
    disable_goal = False

    dim_out = dof * (num_basis + 1)  # weights + goal

    # Disable goal if specified
    if disable_goal:
        dim_out -= dof

    if learn_tau:
        dim_out += 1
    if learn_delay:
        dim_out += 1

    return dim_out


def process_loss(loss: list, epoch: int, prefix: str = ''):
    if not isinstance(loss, list):
        raise ValueError(f"Loss should be a list, got {type(loss)}")
    stats = {}
    loss = torch.tensor(loss)
    stats[prefix + 'mean'] = torch.mean(loss).item()
    stats[prefix + 'std'] = torch.std(loss).item()
    stats[prefix + 'max'] = torch.max(loss).item()
    stats[prefix + 'min'] = torch.min(loss).item()
    stats[prefix + 'epoch'] = epoch

    return stats


def train(device='cuda', epochs=10):
    get_wandb_logger(project_name='Traj_transformer',
                     entity_name='x-huang',
                     group='TrajTransformer',
                     name='1120', local_log_dir='')

    mlp_in = 256
    mlp_out = dim_policy_out()

    # Initialize the model
    model = TrajReconstructor(mlp_in=mlp_in, mlp_out=mlp_out, dropout=0.1)
    # Initialize the MP4Transformer
    mp4 = MP4Transformer()
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prepare the dataset
    train_data, test_data = load_data()
    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # ToDevice
    model.to(device)
    model.float()
    model.train()

    # Train the model - FixMe: Record the loss using WandB
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for data in train_loader:
            data = data.to(device).float()
            optimizer.zero_grad()
            param = model(data)
            result = mp4.get_prodmp_results(param)
            position = result["pos"]
            zeros = torch.zeros_like(position, device=device)
            loss = torch.nn.functional.mse_loss(position + data, zeros)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        stat = process_loss(train_loss, epoch, prefix='train_loss_')
        wandb.log(stat)
        # Test
        test_loss = []
        model.eval()
        for data in test_loader:
            data = data.to(device).float()
            param = model(data)
            result = mp4.get_prodmp_results(param)
            position = result["pos"]
            zeros = torch.zeros_like(position, device=device)
            loss = torch.nn.functional.mse_loss(position + data, zeros)
            test_loss.append(loss.item())
        # Log the loss and the epoch
        stat = process_loss(test_loss, epoch, prefix='test_loss_')
        wandb.log(stat)


if __name__ == '__main__':
    train(device='cuda', epochs=200)



