from typing import Dict, List, Optional, Tuple
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


class GaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        use_tanh: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh = use_tanh

        self.network = MLP(
            input_dim=observation_dim,
            output_dim=action_dim * 2,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if observation.dim() != 2:
            raise ValueError(
                f"Expected observation shape (B, obs_dim), got {observation.shape}"
            )

        output = self.network(observation)
        loc, log_std = output.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        scale = torch.exp(log_std)

        return loc, scale

    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, scale = self.forward(observation)

        if deterministic:
            if self.use_tanh:
                action = torch.tanh(loc)
            else:
                action = loc
            log_prob = torch.zeros_like(action)
        else:
            distribution = torch.distributions.Normal(loc, scale)
            if self.use_tanh:
                raw_action = distribution.rsample()
                action = torch.tanh(raw_action)
                log_prob = self._compute_tanh_log_prob(distribution, raw_action, action)
            else:
                action = distribution.rsample()
                log_prob = distribution.log_prob(action)

        return action, log_prob

    def _compute_tanh_log_prob(
        self,
        distribution: torch.distributions.Normal,
        raw_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = distribution.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class DeterministicActor(nn.Module):
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

        self.network = MLP(
            input_dim=observation_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2:
            raise ValueError(
                f"Expected observation shape (B, obs_dim), got {observation.shape}"
            )
        action = self.network(observation)
        return torch.tanh(action)

    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.forward(observation)
        log_prob = torch.zeros_like(action)
        return action, log_prob


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        use_tanh: bool = True,
        deterministic: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh = use_tanh
        self.deterministic = deterministic
        self.dropout = dropout

        if deterministic:
            self._actor = DeterministicActor(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )
        else:
            self._actor = GaussianActor(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                use_tanh=use_tanh,
                dropout=dropout,
            )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.deterministic:
            action = self._actor(observation)
            return {
                "action": action,
                "loc": action,
                "scale": torch.ones_like(action),
            }
        else:
            loc, scale = self._actor(observation)
            return {
                "action": torch.tanh(loc) if self.use_tanh else loc,
                "loc": loc,
                "scale": scale,
            }

    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic is None:
            deterministic = self.deterministic
        return self._actor.get_action(observation, deterministic=deterministic)

    @property
    def is_stochastic(self) -> bool:
        return not self.deterministic

    @property
    def is_deterministic(self) -> bool:
        return self.deterministic