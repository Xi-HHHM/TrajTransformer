import torch
import os

import wandb
import argparse
import yaml
from tqdm import tqdm
import numpy as np

from model import TrajReconstructor
from utils.util_mp import MP4Transformer
from utils.util_data import load_traj_from_hdf5
from utils.util_wandb import get_wandb_logger, plot_reconstruction, log_artifact
from torch.utils.data import DataLoader, Dataset

from fancy_gym.envs.mujoco.sb_planning.sbp_wrapper.collision_checker import CollisionChecker


class WarmupInverseSqrtSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, model_size, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


class CustomDataset(Dataset):
    def __init__(self, data, input_type='joint'):
        if isinstance(data, list):
            if input_type == 'extend':
                # Extend the trajectory by filling the other two columns with other values
                data_dict = {i: self.extend(d) for i, d in enumerate(data)}
            else:
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

    @staticmethod
    def extend(traj):
        extended_traj = np.zeros((traj.shape[0], traj.shape[1] * 3))
        # Extend the trajectory by filling the other two columns with other values
        extended_traj[:, 0:6] = traj
        extended_traj[:, 6:12] = np.sin(traj)
        extended_traj[:, 12:18] = np.cos(traj)
        return extended_traj


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
    stats['epoch'] = epoch

    return stats


def train(config: dict):
    wandb_config = config['wandb']
    project_name = wandb_config['project']
    entity_name = wandb_config['entity']
    group = wandb_config['group']
    name = wandb_config['name']

    train_config = config['train']
    device = train_config['device']
    epochs = train_config['epochs']
    warmup_epochs = train_config['warmup_epochs']
    n_checkpoints = train_config['n_checkpoints']
    lr_scheduler = train_config['lr_scheduler']
    lr_step_size = lr_scheduler['step_size']
    lr_gamma = lr_scheduler['gamma']
    save_interval = int(epochs / n_checkpoints)

    # Hyperparameters
    lr = train_config['lr']
    wd = train_config['wd']
    batch_size = train_config['batch_size']

    # MPs
    model_config = config['model']
    mlp_in = model_config['mp']['mlp']['in_dim']
    mlp_n_layers = model_config['mp']['mlp']['n_layer']
    mlp_n_hidden = model_config['mp']['mlp']['hidden_dim']
    mlp_out = dim_policy_out()

    # Transformer
    transformer_pre_mlp_input_dim = model_config['transformer']['pre_mlp_input_dim']
    transformer_linear_input_type = model_config['transformer']['input_type']
    transformer_emb_dim = model_config['transformer']['emb_dim']
    transformer_n_layers = model_config['transformer']['n_layer']
    transformer_n_heads = model_config['transformer']['n_head']
    transformer_dropout = model_config['transformer']['dropout']
    transformer_register_tokens = model_config['transformer']['register_tokens']
    if transformer_register_tokens is False:
        transformer_register_tokens = 0

    get_wandb_logger(project_name=project_name,
                     entity_name=entity_name,
                     group=group, name=name, local_log_dir='', config=config)

    # Initialize the model
    model = TrajReconstructor(pre_mlp_in=transformer_pre_mlp_input_dim,
                              post_mlp_out=mlp_out,
                              mlp_n_hidden=mlp_n_hidden,
                              mlp_n_layers=mlp_n_layers,
                              transformer_emb_dim=transformer_emb_dim,
                              transformer_depth=transformer_n_layers,
                              transformer_heads=transformer_n_heads,
                              transformer_register_tokens=transformer_register_tokens,
                              dropout=transformer_dropout)
    # Initialize the MP4Transformer
    mp4 = MP4Transformer(device=device)
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = WarmupInverseSqrtSchedule(optimizer, warmup_steps=warmup_epochs,
    # model_size=transformer_emb_dim)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Prepare the dataset
    train_data, test_data = load_data()
    train_dataset = CustomDataset(train_data, input_type=transformer_linear_input_type)
    test_dataset = CustomDataset(test_data, input_type=transformer_linear_input_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ToDevice
    model.to(device)
    model.float()
    model.train()
    progress_bar = tqdm(total=epochs, desc='Training')

    # Train the model - FixMe: Record the loss using WandB
    for epoch in range(epochs):
        train_loss = []
        diffs = []
        model.train()
        for data in train_loader:
            data = data.to(device).float()
            optimizer.zero_grad()
            param = model(data)
            result = mp4.get_prodmp_results(param, data)
            position = result["pos"]
            zeros = torch.zeros_like(position, device=device)
            loss = torch.nn.functional.mse_loss(position - data[..., :6], zeros)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            diff = position - data[..., :6]
            # Find out the max l2 difference in sequence
            diff.detach()
            diff = torch.norm(diff, p=2, dim=2)
            diff_max = torch.max(diff)
            diffs.append(diff_max.item())

        stat = process_loss(train_loss, epoch, prefix='train_loss_')
        stat_diff = process_loss(diffs, epoch, prefix='train_max_diff_')
        stat.update(stat_diff)
        wandb.log(stat)
        # Test
        test_loss = []
        diffs = []
        model.eval()
        for data in test_loader:
            data = data.to(device).float()
            param = model(data)
            result = mp4.get_prodmp_results(param, data)
            position = result["pos"]
            zeros = torch.zeros_like(position, device=device)
            loss = torch.nn.functional.mse_loss(position - data[..., :6], zeros)
            test_loss.append(loss.item())
            diff = position - data[..., :6]
            diff.detach()
            # Find out the max l2 difference in sequence
            diff = torch.norm(diff, p=2, dim=2)  # B x T
            diff_max = torch.max(diff)
            diff_max_index = torch.argmax(diff)
            diffs.append(diff_max.item())

        # Log the loss and the epoch
        stat = process_loss(test_loss, epoch, prefix='test_loss_')
        stat_diff = process_loss(diffs, epoch, prefix='test_max_diff_')
        stat.update(stat_diff)
        wandb.log(stat)
        wandb.log({'lr': scheduler.get_lr()[0]})
        scheduler.step()
        progress_bar.update(1)

        if (epoch + 1) % save_interval == 0:
            torch.save(model.encoder.state_dict(),
                       os.path.join(wandb.run.dir, f"TrajReconstructorEncoder_{epoch}.pt"))
            torch.save(model.fc.state_dict(),
                       os.path.join(wandb.run.dir, f"TrajReconstructorFC_{epoch}.pt"))

            traj_index = diff_max_index // 51
            plot_reconstruction(data[traj_index], position[traj_index], epoch)
            wandb.save(os.path.join(wandb.run.dir, "*.pt"))
        log_artifact()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to the config file')
    args = parser.parse_args()
    cfg_file = args.cfg
    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.FullLoader)

    train(cfg)



