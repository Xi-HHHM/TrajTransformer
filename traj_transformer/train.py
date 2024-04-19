import torch
import os
from model import TrajReconstructor
from utils.util_mp import MP4Transformer
from utils.util_data import load_traj_from_hdf5
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, list):
            data_dict = {i: d for i, d in enumerate(data)}
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
    # Get all hdf5 files in the data directory
    files = [f for f in os.listdir('data') if f.endswith('.h5')]
    data = []
    for file in files:
        file = os.path.join('data', file)
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


def train():
    mlp_in = 256
    mlp_out = dim_policy_out()

    # Initialize the model
    model = TrajReconstructor(mlp_in=mlp_in, mlp_out=mlp_out, dropout=0.1)
    # Initialize the MP4Transformer
    mp4 = MP4Transformer(relative_goal=False, disable_goal=False)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prepare the dataset
    train_data, test_data = load_data()
    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # ToDevice
    model.to('cuda')
    model.float()
    model.train()

    # Train the model - FixMe: Record the loss using WandB
    for epoch in range(10):
        for data in train_loader:
            data = data.to('cuda').float()
            optimizer.zero_grad()
            param = model(data)
            result = mp4.get_prodmp_results(param)
            position = result["pos"]
            zeros = torch.zeros_like(position, device='cuda')
            loss = torch.nn.functional.mse_loss(position + data, zeros)
            loss.backward()
            optimizer.step()
        # Test
        loss_sum = 0
        for data in test_loader:
            data = data.to('cuda').float()
            param = model(data)
            result = mp4.get_prodmp_results(param)
            position = result["pos"]
            zeros = torch.zeros_like(position, device='cuda')
            loss = torch.nn.functional.mse_loss(position + data, zeros)
            loss_sum += loss.item()
        print(f'Epoch {epoch}: Loss {loss_sum / len(test_loader)}')


if __name__ == '__main__':
    train()



