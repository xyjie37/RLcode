from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import torch
from torch import nn, optim

try:
    from tensordict import TensorDict

    TensorDictType = TensorDict
except ImportError:
    TensorDictType = None

BatchType = Union["TensorDict", Dict[str, torch.Tensor]]
Self = TypeVar("Self", bound="OfflinePolicy")


class OfflinePolicy(abc.ABC):
    """
    Abstract base class for offline reinforcement learning algorithms.

    This class defines the common interface and behavior contract that all
    offline RL algorithms (CQL, IQL, DT, etc.) must adhere to. It provides
    device management, model I/O utilities, and enforces implementation of
    core training methods.

    Subclasses should:
        - Implement all @abstractmethod decorated methods
        - Expose main networks via self.actor, self.critic attributes
        - Follow optimizer naming conventions (self.actor_optimizer, etc.)

    Attributes:
        device: Device to place models and data on.
        observation_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._device = torch.device(device)
        self._config = kwargs

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, new_device: Union[str, torch.device]) -> None:
        self._device = torch.device(new_device)
        self._move_models_to_device()

    def _move_models_to_device(self) -> None:
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    attr.to(self._device)
            except (AttributeError, TypeError):
                continue

    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    @property
    def actor(self) -> Optional[nn.Module]:
        if hasattr(self, "_actor"):
            return self._actor
        return None
    @actor.setter
    def actor(self, value: nn.Module) -> None:
        self._actor = value
    @property
    def critic(self) -> Optional[nn.Module]:
        if hasattr(self, "_critic"):
            return self._critic
        return None
    @critic.setter
    def critic(self, value: nn.Module) -> None:
        self._critic = value
    @property
    def actor_optimizer(self) -> Optional[optim.Optimizer]:
        if hasattr(self, "_actor_optimizer"):
            return self._actor_optimizer
        return None

    @property
    def critic_optimizer(self) -> Optional[optim.Optimizer]:
        if hasattr(self, "_critic_optimizer"):
            return self._critic_optimizer
        return None

    @abc.abstractmethod
    def train_step(
        self,
        batch: BatchType,
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Training batch containing observations, actions, rewards,
                   next_observations, and done flags.
                   Can be either a TensorDict or a standard dict.

        Returns:
            Dictionary of training metrics (e.g., q_loss, actor_loss, cql_loss).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select an action based on the current observation.

        Args:
            obs: Observation tensor with shape (obs_dim,) or (B, obs_dim).
            deterministic: Whether to select deterministic action.
                          If True, returns mean/tanh(action). If False, samples.

        Returns:
            Action tensor with shape (act_dim,) or (B, act_dim).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(
        self,
        path: Union[str, Path],
    ) -> None:
        """
        Save the policy state to a file.

        Args:
            path: File path to save the policy state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(
        self,
        path: Union[str, Path],
    ) -> None:
        """
        Load the policy state from a file.

        Args:
            path: File path to load the policy state from.
        """
        raise NotImplementedError

    def _get_save_dict(
        self,
        include_optimizers: bool = True,
    ) -> Dict[str, Any]:
        save_dict = {
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "config": self._config,
        }

        if self.actor is not None:
            save_dict["actor_state_dict"] = self.actor.state_dict()

        if self.critic is not None:
            save_dict["critic_state_dict"] = self.critic.state_dict()

        if include_optimizers:
            if self.actor_optimizer is not None:
                save_dict["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            if self.critic_optimizer is not None:
                save_dict["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()

        return save_dict

    def _load_from_dict(
        self,
        save_dict: Dict[str, Any],
        load_optimizers: bool = True,
        strict: bool = True,
    ) -> None:
        if "observation_dim" in save_dict:
            self._observation_dim = save_dict["observation_dim"]
        if "action_dim" in save_dict:
            self._action_dim = save_dict["action_dim"]
        if "config" in save_dict:
            self._config = save_dict["config"]

        if "actor_state_dict" in save_dict and self.actor is not None:
            self.actor.load_state_dict(save_dict["actor_state_dict"], strict=strict)

        if "critic_state_dict" in save_dict and self.critic is not None:
            self.critic.load_state_dict(save_dict["critic_state_dict"], strict=strict)

        if load_optimizers:
            if "actor_optimizer_state_dict" in save_dict and self.actor_optimizer is not None:
                self.actor_optimizer.load_state_dict(save_dict["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in save_dict and self.critic_optimizer is not None:
                self.critic_optimizer.load_state_dict(save_dict["critic_optimizer_state_dict"])

    def eval(self) -> None:
        if self.actor is not None:
            self.actor.eval()
        if self.critic is not None:
            self.critic.eval()

    def train(self, mode: bool = True) -> Self:
        if self.actor is not None:
            self.actor.train(mode)
        if self.critic is not None:
            self.critic.train(mode)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_dim={self._observation_dim}, "
            f"act_dim={self._action_dim}, "
            f"device={self._device})"
        )