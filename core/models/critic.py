from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        input_dim = observation_dim + action_dim
        self.network = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if observation.dim() != 2:
            raise ValueError(
                f"Expected observation shape (B, obs_dim), got {observation.shape}"
            )
        if action.dim() != 2:
            raise ValueError(
                f"Expected action shape (B, act_dim), got {action.shape}"
            )
        if observation.shape[0] != action.shape[0]:
            raise ValueError(
                f"Batch size mismatch: observation batch={observation.shape[0]}, "
                f"action batch={action.shape[0]}"
            )

        obs_action = torch.cat([observation, action], dim=-1)
        q_value = self.network(obs_action)
        return q_value


class DoubleQCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        num_q_nets: int = 2,
    ):
        super().__init__()
        if num_q_nets < 1:
            raise ValueError(f"num_q_nets must be >= 1, got {num_q_nets}")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.num_q_nets = num_q_nets

        self.q_nets = nn.ModuleList([
            QNetwork(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_q_nets)
        ])

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if observation.dim() != 2:
            raise ValueError(
                f"Expected observation shape (B, obs_dim), got {observation.shape}"
            )
        if action.dim() != 2:
            raise ValueError(
                f"Expected action shape (B, act_dim), got {action.shape}"
            )

        q_values = tuple(q_net(observation, action) for q_net in self.q_nets)
        return q_values

    def get_q_values(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.forward(observation, action)
        return torch.cat(q_values, dim=-1)

    def get_min_q(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(observation, action)
        return torch.minimum(*q_values)

    def get_max_q(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(observation, action)
        return torch.maximum(*q_values)


class EnsembleQCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        num_q_nets: int = 5,
    ):
        super().__init__()
        if num_q_nets < 2:
            raise ValueError(f"EnsembleQCritic requires num_q_nets >= 2, got {num_q_nets}")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.num_q_nets = num_q_nets

        self.q_nets = nn.ModuleList([
            QNetwork(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_q_nets)
        ])

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if observation.dim() != 2:
            raise ValueError(
                f"Expected observation shape (B, obs_dim), got {observation.shape}"
            )
        if action.dim() != 2:
            raise ValueError(
                f"Expected action shape (B, act_dim), got {action.shape}"
            )

        q_values = tuple(q_net(observation, action) for q_net in self.q_nets)
        return q_values

    def get_q_values(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.forward(observation, action)
        return torch.cat(q_values, dim=-1)

    def get_statistics(self, observation: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_values = self.get_q_values(observation, action)
        return {
            "mean": q_values.mean(dim=-1, keepdim=True),
            "std": q_values.std(dim=-1, keepdim=True),
            "min": q_values.min(dim=-1, keepdim=True)[0],
            "max": q_values.max(dim=-1, keepdim=True)[0],
        }


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        num_q_nets: int = 2,
        ensemble: bool = False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.num_q_nets = num_q_nets
        self.ensemble = ensemble

        if ensemble:
            self._critic = EnsembleQCritic(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                num_q_nets=num_q_nets,
            )
        else:
            self._critic = DoubleQCritic(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                num_q_nets=num_q_nets,
            )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        return self._critic.forward(observation, action)

    def get_q_values(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self._critic.get_q_values(observation, action)

    def get_min_q(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if hasattr(self._critic, "get_min_q"):
            return self._critic.get_min_q(observation, action)
        q_values = self.forward(observation, action)
        return torch.minimum(*q_values)

    def get_max_q(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if hasattr(self._critic, "get_max_q"):
            return self._critic.get_max_q(observation, action)
        q_values = self.forward(observation, action)
        return torch.maximum(*q_values)

    def get_statistics(self, observation: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hasattr(self._critic, "get_statistics"):
            return self._critic.get_statistics(observation, action)
        q_values = self.get_q_values(observation, action)
        return {
            "mean": q_values.mean(dim=-1, keepdim=True),
            "std": q_values.std(dim=-1, keepdim=True),
            "min": q_values.min(dim=-1, keepdim=True)[0],
            "max": q_values.max(dim=-1, keepdim=True)[0],
        }

    @property
    def is_ensemble(self) -> bool:
        return self.ensemble

    @property
    def num_networks(self) -> int:
        return self.num_q_nets