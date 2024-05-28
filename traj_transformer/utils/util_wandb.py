import wandb
import torch
import os
import matplotlib.pyplot as plt


def get_wandb_logger(project_name, entity_name, group, name, local_log_dir, config):
    wandb.init(project=project_name, entity=entity_name, group=group, name=name,
                     dir=local_log_dir, config=config)


def log_artifact():
    artifact = wandb.Artifact(name="model", type="model")
    artifact.add_dir(local_path=wandb.run.dir)  # Add dataset directory to artifact
    wandb.log_artifact(artifact)


def download_artifact(model_name):
    artifact = wandb.use_artifact(model_name, type="model")
    artifact_dir = artifact.download(root=wandb.run.dir)
    return artifact_dir


def plot_reconstruction(data, recon, epoch, show=False):
    """
    Plot the reference and actual robot trajectories
    """
    # Check if the data is tensor or numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(recon, torch.Tensor):
        recon = recon.detach().cpu().numpy()
        if recon.ndim == 3:
            recon = recon.squeeze(0)

    fig, axs = plt.subplots(6, 1, figsize=(12, 8))

    for i in range(6):
        # Reference trajectory
        ref_traj = data[:, i]

        # Trajectory that robot actually follows
        actual_traj = recon[:, i]

        # diff = np.abs(ref_traj - actual_traj)
        axs[i].plot(ref_traj, label="ground_truth", color="green")
        axs[i].plot(actual_traj, label="reconstruction", color="blue")
        axs[i].legend()
        axs[i].set_title(f"Joint {i}")

    # Save to wandb folder
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(wandb.run.dir, f"reconstruction_{epoch}.png"))

