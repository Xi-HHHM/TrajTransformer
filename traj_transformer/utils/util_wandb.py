import wandb


def get_wandb_logger(project_name, entity_name, group, name, local_log_dir):
    wandb.init(project=project_name, entity=entity_name, group=group, name=name,
               dir=local_log_dir)


def save_model_to_wandb(save_dir):
    wandb.save(save_dir)

