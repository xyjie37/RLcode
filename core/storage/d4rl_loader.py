from typing import Dict, Optional, Tuple
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple, Union


def check_d4rl_available() -> bool:
    try:
        import d4rl
        return True
    except ImportError:
        return False


def get_normalization_stats(
    observations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = observations.mean(axis=0)
    std = observations.std(axis=0) + 1e-6
    return mean, std


def normalize_observations(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (observations - mean) / std


class D4RLDataset:
    def __init__(
        self,
        env_name: str,
        normalize_obs: bool = False,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        if not check_d4rl_available():
            raise ImportError(
                "D4RL is not installed. Please install it with: pip install d4rl"
            )

        import d4rl
        import gym

        try:
            env = gym.make(env_name)
        except Exception as e:
            raise ValueError(f"Failed to create environment '{env_name}': {e}")

        try:
            dataset_dict = d4rl.qlearning_dataset(env)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset for '{env_name}': {e}")
        finally:
            env.close()

        self.env_name = env_name
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        observations = dataset_dict["observations"]
        actions = dataset_dict["actions"]
        rewards = dataset_dict["rewards"]
        next_observations = dataset_dict["next_observations"]
        terminals = dataset_dict.get("terminals", np.zeros(len(observations), dtype=np.bool_))
        timeouts = dataset_dict.get("timeouts", np.zeros(len(observations), dtype=np.bool_))

        self.num_samples = len(observations)
        self.normalize_obs = normalize_obs

        if normalize_obs:
            if obs_mean is None or obs_std is None:
                self.obs_mean, self.obs_std = get_normalization_stats(observations)
            else:
                self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
                self.obs_std = np.asarray(obs_std, dtype=np.float32)
            observations = normalize_observations(observations, self.obs_mean, self.obs_std)
            next_observations = normalize_observations(next_observations, self.obs_mean, self.obs_std)
        else:
            self.obs_mean = np.zeros(self.observation_dim, dtype=np.float32)
            self.obs_std = np.ones(self.observation_dim, dtype=np.float32)

        self.observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1, 1)
        self.next_observations = torch.as_tensor(next_observations, dtype=torch.float32, device=device)
        self.terminals = torch.as_tensor(terminals, dtype=torch.bool, device=device).reshape(-1, 1)
        self.timeouts = torch.as_tensor(timeouts, dtype=torch.bool, device=device).reshape(-1, 1)
        self.dones = (terminals | timeouts).reshape(-1, 1)

        self._device = device

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.observations[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_observations[idx],
            "done": self.dones[idx],
            "terminal": self.terminals[idx],
        }

    def sample(
        self,
        batch_size: int,
        replace: bool = True,
    ) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(
            self.num_samples,
            size=batch_size,
            replace=replace,
        )

        return {
            "obs": self.observations[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "done": self.dones[indices],
            "terminal": self.terminals[indices],
        }

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.observations,
            "action": self.actions,
            "reward": self.rewards,
            "next_obs": self.next_observations,
            "done": self.dones,
            "terminal": self.terminals,
        }

    def to(self, device: Union[str, torch.device]) -> "D4RLDataset":
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_observations = self.next_observations.to(device)
        self.terminals = self.terminals.to(device)
        self.timeouts = self.timeouts.to(device)
        self.dones = self.dones.to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return torch.device(self._device) if isinstance(self._device, str) else self._device


def load_d4rl_dataset(
    env_name: str,
    normalize_obs: bool = False,
    obs_mean: Optional[np.ndarray] = None,
    obs_std: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cpu",
    use_tensordict: bool = False,
) -> Union[Dict[str, torch.Tensor], "tensordict.TensorDict"]:
    dataset = D4RLDataset(
        env_name=env_name,
        normalize_obs=normalize_obs,
        obs_mean=obs_mean,
        obs_std=obs_std,
        device=device,
    )

    if use_tensordict:
        try:
            from tensordict import TensorDict
            return TensorDict({
                "obs": dataset.observations,
                "action": dataset.actions,
                "reward": dataset.rewards,
                "next_obs": dataset.next_observations,
                "done": dataset.dones,
                "terminal": dataset.terminals,
            }, batch_size=[dataset.num_samples])
        except ImportError:
            import warnings
            warnings.warn(
                "tensordict not available, falling back to dict",
                RuntimeWarning,
            )
            return dataset.get_all_data()
    else:
        return dataset.get_all_data()