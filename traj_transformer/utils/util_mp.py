import torch
from addict import Dict

from mp_pytorch import util
from mp_pytorch.mp import MPFactory


class MP4Transformer:
    def __init__(self, relative_goal=False, disable_goal=False):
        config, times, init_time, init_pos, init_vel = self.get_mp_config(
            relative_goal, disable_goal)

        self.config = config
        self.times = times.to("cuda")
        self.init_time = init_time.to("cuda")
        self.init_pos = init_pos.to("cuda")
        self.init_vel = init_vel.to("cuda")

        self.mp = MPFactory.init_mp(**config)

    @staticmethod
    def get_mp_config(relative_goal=False, disable_goal=False):
        """
        Get the config of DMPs for testing

        Args:
            relative_goal: if True, the goal is relative to the initial position
            disable_goal:

        Returns:
            config in dictionary
        """

        torch.manual_seed(0)

        config = Dict()
        config.device = "cuda"
        config.mp_type = "prodmp"
        config.num_dof = 6
        config.tau = 5
        config.learn_tau = False
        config.learn_delay = False

        config.mp_args.num_basis = 3
        config.mp_args.basis_bandwidth_factor = 2
        config.mp_args.num_basis_outside = 0
        config.mp_args.alpha = 25
        config.mp_args.alpha_phase = 2
        config.mp_args.dt = 0.1
        config.mp_args.relative_goal = relative_goal
        config.mp_args.disable_goal = disable_goal
        config.mp_args.weights_scale = torch.ones([3], device="cuda") * 0.5
        config.mp_args.goal_scale = 0.5

        # assume we have 64 trajectories in a batch
        num_traj = 64

        # Get trajectory scaling
        tau, delay = 4, 1
        scale_delay = torch.Tensor([tau, delay])
        scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

        # Get times
        num_t = int(config.tau / config.mp_args.dt) + 1
        times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
        times = util.add_expand_dim(times, [0], [num_traj])

        # Get IC
        init_time = times[:, 0]
        init_pos_scalar = 1
        init_pos = torch.zeros([num_traj, config.num_dof]) # init_pos_scalar * torch.ones([num_traj, config.num_dof])
        init_vel = torch.zeros_like(init_pos)

        # Get params
        goal = init_pos_scalar
        if relative_goal:
            goal -= init_pos_scalar

        return config, times, init_time, init_pos, init_vel

    def get_prodmp_results(self, params):
        self.mp.update_inputs(self.times, params, None, self.init_time, self.init_pos, self.init_vel)
        result_dict = self.mp.get_trajs()
        return result_dict

