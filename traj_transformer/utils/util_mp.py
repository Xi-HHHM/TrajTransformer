import torch
from addict import Dict

from mp_pytorch import util
from mp_pytorch.mp import MPFactory

from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator


class MP4Transformer:
    def __init__(self):
        self.duration = 5
        self.dt = 0.1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.traj_gen = self.get_traj_gen()

    def get_traj_gen(self):
        kwargs_dict_ur_prodmp = {
            "name": 'EnvName',
            "wrappers": [],
            "trajectory_generator_kwargs": {
                'trajectory_generator_type': 'prodmp',
                'duration': 2.0,
                'weights_scale': 1.0,
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'exp',
                'tau': 1.5,
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": 1.0,
                "d_gains": 0.1,
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'prodmp',
                'alpha': 10,
                'num_basis': 5,
            },
            "black_box_kwargs": {
            }
        }

        kwargs_dict_ur_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.7
        kwargs_dict_ur_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.5
        kwargs_dict_ur_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
        kwargs_dict_ur_prodmp['trajectory_generator_kwargs']['relative_goal'] = False
        kwargs_dict_ur_prodmp['trajectory_generator_kwargs']['disable_goal'] = False
        kwargs_dict_ur_prodmp['basis_generator_kwargs']['num_basis'] = 3
        kwargs_dict_ur_prodmp['phase_generator_kwargs']['tau'] = 5
        kwargs_dict_ur_prodmp['phase_generator_kwargs']['learn_tau'] = False
        kwargs_dict_ur_prodmp['phase_generator_kwargs']['learn_delay'] = False
        kwargs_dict_ur_prodmp['basis_generator_kwargs']['alpha'] = 25.
        kwargs_dict_ur_prodmp['basis_generator_kwargs']['basis_bandwidth_factor'] = 1
        kwargs_dict_ur_prodmp['phase_generator_kwargs']['alpha_phase'] = 3

        traj_gen_kwargs = kwargs_dict_ur_prodmp.pop("trajectory_generator_kwargs", {})
        black_box_kwargs = kwargs_dict_ur_prodmp.pop('black_box_kwargs', {})
        phase_kwargs = kwargs_dict_ur_prodmp.pop("phase_generator_kwargs", {})
        basis_kwargs = kwargs_dict_ur_prodmp.pop("basis_generator_kwargs", {})

        traj_gen_kwargs['device'] = self.device
        phase_kwargs['device'] = self.device
        basis_kwargs['device'] = self.device

        # Fixme: not sure about this
        traj_gen_kwargs['action_dim'] = 6

        if black_box_kwargs.get('duration') is None:
            black_box_kwargs['duration'] = self.duration
        if phase_kwargs.get('tau') is None:
            phase_kwargs['tau'] = black_box_kwargs['duration']

        phase_gen = get_phase_generator(**phase_kwargs)
        basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_kwargs)
        traj_gen = get_trajectory_generator(basis_generator=basis_gen, **traj_gen_kwargs)
        traj_gen.set_duration(self.duration, self.dt)

        return traj_gen

    def get_prodmp_results(self, params, data):
        batch_size = params.shape[0]
        self.traj_gen.set_params(params)

        condition_pos = data[:, 0, :]
        condition_vel = torch.zeros([batch_size, 6], device=params.device)
        init_time = torch.zeros([batch_size], device=params.device)

        try:
            self.traj_gen.set_initial_conditions(init_time, condition_pos, condition_vel)
            self.traj_gen.set_duration(self.duration, self.dt, include_init_time=True)
        except Exception as e:
            print("self.traj_gen.basis_gn.pre_compute_length_factor", self.traj_gen.basis_gn.pre_compute_length_factor)
            print("init_time.max()", init_time.max())
            print("init_time.max() > self.traj_gen.basis_gn.pre_compute_length_factor",
                  init_time.max() > self.traj_gen.basis_gn.pre_compute_length_factor)
            raise e

        result_dict = self.traj_gen.get_trajs()
        return result_dict


class MP4TransformerDeprecated:
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
        tau, delay = 5, 0
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


if __name__ == "__main__":
    mp4 = MP4Transformer()
    params = torch.randn([64, 24], device="cpu")
    result = mp4.get_prodmp_results(params)
    print(result['pos'].size())
