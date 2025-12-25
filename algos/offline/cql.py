from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union

from core.models.actor import GaussianActor
from core.models.critic import QNetwork
from algos.base.offline_policy import OfflinePolicy

from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,  # noqa: F401
)
from torchrl.data import Bounded
from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class CQL(OfflinePolicy):
    """
    CQL (Conservative Q-Learning) Offline RL
    基于官方实现的完整 CQL 算法
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor: Dict[str, Any],
        critic: Dict[str, Any],
        cql_alpha: float = 5.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        use_automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = "cpu",
        max_action: float = 1.0,
        num_random_actions: int = 10,
        temperature: float = 1.0,
        alpha_lr: float = 3e-4,
        target_update_eps: float = 5e-3,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            device=device,
        )

        self._cfg = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "actor": actor,
            "critic": critic,
            "cql_alpha": cql_alpha,
            "tau": tau,
            "gamma": gamma,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "use_automatic_entropy_tuning": use_automatic_entropy_tuning,
            "target_entropy": target_entropy,
            "device": device,
            "max_action": max_action,
            "num_random_actions": num_random_actions,
            "temperature": temperature,
            "alpha_lr": alpha_lr,
            "target_update_eps": target_update_eps,
            "dropout": dropout,
        }

        # 超参数
        self._tau = tau if tau is not None else target_update_eps
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.max_action = max_action
        self.num_random_actions = num_random_actions
        self.temperature = temperature
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        # 目标熵初始化
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        # 熵温度参数alpha
        if self.use_automatic_entropy_tuning:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(np.log(cql_alpha), dtype=torch.float32, device=device)
            )
            self._alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cql_alpha), dtype=torch.float32, device=device)
            self.log_alpha.requires_grad_(False)
            self._alpha_optimizer = None

        # 动作空间配置
        self.action_spec = Bounded(
            low=-self.max_action,
            high=self.max_action,
            shape=(action_dim,),
            dtype=torch.float32,
        )

        # 构建Actor网络（输出 loc, scale）
        self.gaussian_actor = GaussianActor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=actor.get("hidden_dims", [256, 256]),
            activation=actor.get("activation", "relu"),
            log_std_min=actor.get("log_std_min", -5.0),
            log_std_max=actor.get("log_std_max", 2.0),
            use_tanh=False,
            dropout=dropout,
        ).to(device)

        _actor_module = TensorDictModule(
            self.gaussian_actor,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        self.actor = ProbabilisticActor(
            module=_actor_module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        ).to(device)

        # 构建当前Q网络
        self.q1_net = TensorDictModule(
            QNetwork(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=critic.get("hidden_dims", [256, 256]),
                activation=critic.get("activation", "relu"),
                dropout=dropout,
            ).to(device),
            in_keys=["observation", "action"],
            out_keys=["q_value"],
        )

        self.q2_net = TensorDictModule(
            QNetwork(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=critic.get("hidden_dims", [256, 256]),
                activation=critic.get("activation", "relu"),
                dropout=dropout,
            ).to(device),
            in_keys=["observation", "action"],
            out_keys=["q_value"],
        )

        # 构建目标Q网络
        self.target_q1_net = TensorDictModule(
            QNetwork(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=critic.get("hidden_dims", [256, 256]),
                activation=critic.get("activation", "relu"),
                dropout=dropout,
            ).to(device),
            in_keys=["observation", "action"],
            out_keys=["q_value"],
        )

        self.target_q2_net = TensorDictModule(
            QNetwork(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=critic.get("hidden_dims", [256, 256]),
                activation=critic.get("activation", "relu"),
                dropout=dropout,
            ).to(device),
            in_keys=["observation", "action"],
            out_keys=["q_value"],
        )

        # 初始化目标Q网络参数
        self.target_q1_net.load_state_dict(self.q1_net.state_dict())
        self.target_q2_net.load_state_dict(self.q2_net.state_dict())

        # 冻结目标Q网络参数
        for param in self.target_q1_net.parameters():
            param.requires_grad_(False)
        for param in self.target_q2_net.parameters():
            param.requires_grad_(False)

        # 优化器
        self._actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self._critic_optimizer = optim.Adam(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()),
            lr=critic_lr,
        )

        self._move_models_to_device()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _move_models_to_device(self) -> None:
        """将所有模型移至目标设备"""
        self.actor.to(self.device)
        self.q1_net.to(self.device)
        self.q2_net.to(self.device)
        self.target_q1_net.to(self.device)
        self.target_q2_net.to(self.device)
        if self.use_automatic_entropy_tuning:
            self.log_alpha = self.log_alpha.to(self.device)

    def _soft_update_target_networks(self) -> None:
        """软更新目标网络"""
        for target_param, param in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)
        for target_param, param in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)

    def _compute_conservative_penalty(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        action_dim = self.action_dim

        # 1. 随机动作
        random_actions = (
            torch.rand(batch_size, self.num_random_actions, action_dim, device=obs.device)
            * 2 * self.max_action - self.max_action
        )
        obs_expanded = obs.unsqueeze(1).repeat(1, self.num_random_actions, 1)
        obs_flat = obs_expanded.reshape(-1, obs.shape[1])
        random_actions_flat = random_actions.reshape(-1, action_dim)

        # ✅ 修复：明确 batch_size
        rand_td = TensorDict(
            {"observation": obs_flat, "action": random_actions_flat}, 
            batch_size=[batch_size * self.num_random_actions]  # 👈 关键修改
        )
        q1_rand = self.q1_net(rand_td)["q_value"].reshape(batch_size, self.num_random_actions)
        q2_rand = self.q2_net(rand_td)["q_value"].reshape(batch_size, self.num_random_actions)

        # 2. 策略动作（当前状态）
        actor_td = TensorDict({"observation": obs}, batch_size=[batch_size])
        self.actor(actor_td)
        policy_actions = actor_td["action"]
        
        # ✅ 修复：明确 batch_size
        policy_td = TensorDict(
            {"observation": obs, "action": policy_actions}, 
            batch_size=[batch_size]  # 👈 关键修改
        )
        q1_pi = self.q1_net(policy_td)["q_value"]
        q2_pi = self.q2_net(policy_td)["q_value"]

        # 3. 合并（每个Q网络分别处理）
        q1_cat = torch.cat([q1_rand, q1_pi], dim=1)  # [B, N+1]
        q2_cat = torch.cat([q2_rand, q2_pi], dim=1)  # [B, N+1]

        # 4. 分别计算logsumexp
        logsumexp_q1 = torch.logsumexp(q1_cat / self.temperature, dim=1) * self.temperature
        logsumexp_q2 = torch.logsumexp(q2_cat / self.temperature, dim=1) * self.temperature

        # 取平均
        logsumexp_q = (logsumexp_q1 + logsumexp_q2) / 2.0
        return logsumexp_q
    def train_step(self, batch: Union[TensorDict, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        # 处理输入批次
        if isinstance(batch, dict):
            key_mapping = {
                "obs": "observation",
                "next_obs": "next_observation",
                "act": "action",
                "rew": "reward",
                "done": "done",
            }
            batch_mapped: Dict[str, torch.Tensor] = {}
            for k, v in batch.items():
                target_k = key_mapping.get(k, k)
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v).float()
                elif not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, dtype=torch.float32)
                batch_mapped[target_k] = v.to(self.device)
            batch = TensorDict(batch_mapped, batch_size=[batch_mapped["observation"].shape[0]])
        else:
            batch = batch.to(self.device, non_blocking=True)

        obs = batch["observation"]
        action = batch["action"]
        next_obs = batch["next_observation"]
        reward = batch["reward"].squeeze(-1)
        done = batch["done"].squeeze(-1).float()
        batch_size = obs.shape[0]

        # ========== 1. 更新 Critic ==========
        # 当前 Q(s,a)
        current_td = TensorDict({"observation": obs, "action": action}, batch_size=[batch_size])
        q1_pred = self.q1_net(current_td)["q_value"].squeeze(-1)
        q2_pred = self.q2_net(current_td)["q_value"].squeeze(-1)

        # 目标 Q: r + gamma * min(Q_target(s', pi(s')))
        with torch.no_grad():
            next_actor_td = TensorDict({"observation": next_obs}, batch_size=[batch_size])
            self.actor(next_actor_td)
            next_action = next_actor_td["action"]

            target_td = TensorDict({"observation": next_obs, "action": next_action}, batch_size=[batch_size])
            target_q1 = self.target_q1_net(target_td)["q_value"].squeeze(-1)
            target_q2 = self.target_q2_net(target_td)["q_value"].squeeze(-1)
            target_q = torch.min(target_q1, target_q2)
            y = reward + (1 - done) * self.gamma * target_q

        # Bellman MSE
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)
        q_loss = q1_loss + q2_loss

        # CQL conservative penalty（完整版本）
        logsumexp_q = self._compute_conservative_penalty(obs)  # ✅ 新代码
        q_data = torch.min(q1_pred, q2_pred)
        cql_penalty = (logsumexp_q - q_data).mean()
        
        # 总 Critic loss
        total_critic_loss = q_loss + self.cql_alpha * cql_penalty

        self._critic_optimizer.zero_grad()
        total_critic_loss.backward()
        # 添加梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), max_norm=10.0)
        self._critic_optimizer.step()

        # ========== 2. 更新 Actor ==========
        actor_td = TensorDict({"observation": obs}, batch_size=[batch_size])
        self.actor(actor_td)
        actor_action = actor_td["action"]
        actor_log_prob = actor_td["action_log_prob"].squeeze(-1)

        actor_q_td = TensorDict({"observation": obs, "action": actor_action}, batch_size=[batch_size])
        actor_q1 = self.q1_net(actor_q_td)["q_value"].squeeze(-1)
        actor_q2 = self.q2_net(actor_q_td)["q_value"].squeeze(-1)
        actor_q = torch.min(actor_q1, actor_q2)
        actor_loss = (-actor_q + self.alpha * actor_log_prob).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # ========== 3. 更新 Alpha ==========
        if self.use_automatic_entropy_tuning and self._alpha_optimizer is not None:
            actor_td_alpha = TensorDict({"observation": obs}, batch_size=[batch_size])
            self.actor(actor_td_alpha)
            actor_log_prob_alpha = actor_td_alpha["action_log_prob"].squeeze(-1)
            
            alpha_loss = -(self.log_alpha * (actor_log_prob_alpha + self.target_entropy).detach()).mean()
            
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # 软更新目标网络
        self._soft_update_target_networks()

        # 指标输出
        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "total_critic_loss": total_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "target_q_mean": target_q.mean().item(),
        }
        return metrics

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        with torch.no_grad():
            td = TensorDict({"observation": obs}, batch_size=[1])
            self.actor(td)
            if deterministic:
                action = td.get("loc")
            else:
                action = td.get("action")

            if action is None:
                raise RuntimeError(
                    "Failed to extract action from actor output. "
                    "Expected 'action' or 'loc' key in TensorDict."
                )

            action = action.squeeze(0).cpu()
            if action.dim() == 0:
                action = action.unsqueeze(0)

        return action

    def get_action_with_log_prob(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        with torch.no_grad():
            td = TensorDict({"observation": obs}, batch_size=[1])
            self.actor(td)
            if deterministic:
                action = td.get("loc")
            else:
                action = td.get("action")

            log_prob = td.get("action_log_prob")
            if log_prob is None:
                log_prob = torch.zeros_like(action)

            action = action.squeeze(0).cpu()
            log_prob = log_prob.squeeze(0).cpu()

        return action, log_prob

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "gaussian_actor_state_dict": self.gaussian_actor.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "q1_net_state_dict": self.q1_net.state_dict(),
            "q2_net_state_dict": self.q2_net.state_dict(),
            "target_q1_net_state_dict": self.target_q1_net.state_dict(),
            "target_q2_net_state_dict": self.target_q2_net.state_dict(),
            "actor_optimizer_state_dict": self._actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": self._alpha_optimizer.state_dict()
            if self._alpha_optimizer is not None
            else None,
            "cfg": self._cfg,
        }
        torch.save(save_dict, path)

    def load(self, path: Union[str, Path], strict: bool = True) -> None:
        path = Path(path)
        save_dict = torch.load(path, map_location=self.device)

        self.gaussian_actor.load_state_dict(save_dict["gaussian_actor_state_dict"], strict=strict)
        self.actor.load_state_dict(save_dict["actor_state_dict"], strict=strict)
        self.q1_net.load_state_dict(save_dict["q1_net_state_dict"], strict=strict)
        self.q2_net.load_state_dict(save_dict["q2_net_state_dict"], strict=strict)
        self.target_q1_net.load_state_dict(save_dict["target_q1_net_state_dict"], strict=strict)
        self.target_q2_net.load_state_dict(save_dict["target_q2_net_state_dict"], strict=strict)
        self._actor_optimizer.load_state_dict(save_dict["actor_optimizer_state_dict"])
        self._critic_optimizer.load_state_dict(save_dict["critic_optimizer_state_dict"])

        self.log_alpha = save_dict["log_alpha"].to(self.device)
        if self.use_automatic_entropy_tuning and self._alpha_optimizer is not None:
            self._alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state_dict"])

        self._cfg = save_dict["cfg"]

    def eval(self) -> None:
        self.actor.eval()
        self.q1_net.eval()
        self.q2_net.eval()
        self.target_q1_net.eval()
        self.target_q2_net.eval()

    def train(self, mode: bool = True) -> "CQL":
        self.actor.train(mode)
        self.q1_net.train(mode)
        self.q2_net.train(mode)
        self.target_q1_net.train(mode)
        self.target_q2_net.train(mode)
        return self

    def __repr__(self) -> str:
        return (
            f"CQL("
            f"obs_dim={self.observation_dim}, "
            f"act_dim={self.action_dim}, "
            f"device={self.device}, "
            f"cql_alpha={self.cql_alpha})"
        )