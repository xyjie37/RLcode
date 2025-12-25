from __future__ import annotations

import os
import sys
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize_config_dir
from hydra.core.utils import setup_globals
import hydra

import random
import datetime
import numpy as np
import torch
import gym
from tensordict import TensorDict

from core.storage.d4rl_loader import D4RLDataset, load_d4rl_dataset
from core.utils.seed import set_seed
from algos.base.offline_policy import OfflinePolicy


os.environ["HYDRA_ALLOW_CONFIG_MODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def compute_avg_return(
    env: gym.Env,
    policy: OfflinePolicy,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> float:
    returns = []
    for episode_idx in range(num_episodes):
        obs = env.reset()
        if seed is not None:
            env.seed(seed + episode_idx)
        done = False
        episode_return = 0.0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                # 确保 policy 是对象而不是字典
                if not hasattr(policy, "select_action"):
                    raise RuntimeError(f"Policy object is invalid (type: {type(policy)}). It does not have 'select_action' method.")
                
                action = policy.select_action(obs_tensor, deterministic=True)
                action = action.cpu().numpy()
                if action.shape[0] == 1 and action.ndim > 1:
                    action = action.squeeze(0)

            next_obs, reward, done, info = env.step(action)
            episode_return += reward
            obs = next_obs

        returns.append(episode_return)

    return float(np.mean(returns))


def get_env_specs(env_name: str) -> tuple[int, int]:
    # 在导入 d4rl 之前强制设置环境变量
    import os
    
    # 打印当前环境变量
    print("DEBUG: Before fixing environment variables in get_env_specs")
    
    # 清理和重建 LD_LIBRARY_PATH
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    paths = current_ld.split(':')
    
    # 移除可能干扰的路径（如 opencv 添加的）
    filtered_paths = []
    for path in paths:
        if 'cv2' not in path and 'opencv' not in path and path.strip() != '':
            filtered_paths.append(path)
    
    # 添加必要的路径
    necessary_paths = [
        '/home/jxyrl/.mujoco/mujoco210/bin',
        '/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/nvidia'
    ]
    
    for path in necessary_paths:
        if path not in filtered_paths:
            filtered_paths.append(path)
    
    # 设置新的 LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = ':'.join(filtered_paths)
    
    # 确保其他必要变量已设置
    os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/home/jxyrl/.mujoco/mujoco210'
    
    if 'LD_PRELOAD' not in os.environ:
        os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGLEW.so'
    
    print(f"DEBUG: After fixing LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH')}")
    
    # 现在导入 d4rl
    try:
        import d4rl
    except ImportError as e:
        print(f"Warning: Could not import d4rl: {e}")
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if hasattr(env, "close"):
        env.close()
    return obs_dim, act_dim


def train_one_epoch(
    policy: OfflinePolicy,
    dataset: D4RLDataset,
    batch_size: int,
    step_idx: int,
    log_interval: int,
    epoch: int,
) -> Dict[str, float]:
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)

    epoch_losses = []
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        # batch_indices = indices[batch_start:batch_end] # 未使用，注释掉

        batch = dataset.sample(batch_size=batch_end - batch_start)
        batch = {k: v for k, v in batch.items()}

        loss_dict = policy.train_step(batch)
        epoch_losses.append(loss_dict)

        step_idx += 1

        if step_idx % log_interval == 0:
            avg_loss = {
                k: np.mean([loss[k] for loss in epoch_losses[-log_interval:]])
                for k in epoch_losses[-1].keys()
            }
            log_str = f"Epoch [{epoch}] Step [{step_idx}] "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in avg_loss.items()])
            print(log_str)

    avg_epoch_loss = {}
    if epoch_losses:
        for key in epoch_losses[0].keys():
            avg_epoch_loss[key] = np.mean([loss[key] for loss in epoch_losses])

    return avg_epoch_loss, step_idx


def evaluate(
    policy: OfflinePolicy,
    env_name: str,
    num_episodes: int,
    seed: int,
) -> Dict[str, float]:
    import d4rl # 确保 d4rl 已加载
    env = gym.make(env_name)
    env.seed(seed)

    avg_return = compute_avg_return(env, policy, num_episodes=num_episodes, seed=seed)

    if hasattr(env, "close"):
        env.close()

    return {"eval/return": avg_return}


def save_checkpoint(
    policy: OfflinePolicy,
    output_dir: Path,
    epoch: int,
    step: int,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    policy.save(checkpoint_path)

    return checkpoint_path


def update_actor_spec(policy: OfflinePolicy, obs_dim: int, act_dim: int) -> None:
    from torchrl.data import BoundedTensorSpec

    max_action = 1.0
    action_spec = BoundedTensorSpec(
        minimum=-max_action,
        maximum=max_action,
        shape=(act_dim,),
        dtype=torch.float32,
    )

    if hasattr(policy, "action_spec"):
        policy.action_spec = action_spec


def main(cfg: DictConfig) -> None:
    print("\n" + "=" * 60)
    print("Offline RL Training")
    print("=" * 60)

    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    env_name = cfg.get("env_name", "halfcheetah-medium-v2")
    print(f"Environment: {env_name}")

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    print(f"Random seed: {seed}")
    set_seed(seed)

    num_epochs = training_cfg.get("num_epochs", 1000)
    # num_steps_per_epoch = training_cfg.get("num_steps_per_epoch", 1000) # 未使用
    batch_size = training_cfg.get("batch_size", 256)
    eval_freq = training_cfg.get("eval_freq", 10)
    save_freq = training_cfg.get("save_freq", 100)
    eval_num_episodes = training_cfg.get("eval_num_episodes", 10)
    # warmup_steps = training_cfg.get("warmup_steps", 0) # 未使用

    log_cfg = cfg.get("logging", {})
    log_interval = log_cfg.get("log_interval", 100)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path.cwd() / "outputs" / f"train_offline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    obs_dim, act_dim = get_env_specs(env_name)
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")

    # 允许修改配置
    OmegaConf.set_struct(cfg, False)
    
    # 更新 actor 配置
    if "actor" in cfg:
        if isinstance(cfg.actor, dict) or isinstance(cfg.actor, DictConfig):
            if "observation_dim" not in cfg.actor or cfg.actor.get("observation_dim") == "${actor.obs_dim}":
                cfg.actor["observation_dim"] = obs_dim
            if "action_dim" not in cfg.actor or cfg.actor.get("action_dim") == "${actor.act_dim}":
                cfg.actor["action_dim"] = act_dim

    normalize_obs = cfg.get("normalize_obs", True)

    print("\nLoading dataset...")
    dataset = D4RLDataset(
        env_name=env_name,
        normalize_obs=normalize_obs,
        device=device,
    )
    print(f"Dataset size: {len(dataset)} samples")

    print("\nInitializing algorithm...")
    algo_cfg = cfg.get("algo", {})

    # ==========================================
    # ✅ 关键修复：检查 algo_cfg 是否包含 _target_
    # ==========================================
    if not algo_cfg:
        raise ValueError("Config 'algo' is empty! Please check your yaml file.")

    # 兼容性修复：如果 yaml 里写的是 class 或 target 而不是 _target_
    if "_target_" not in algo_cfg:
        if "class" in algo_cfg:
            print("[WARNING] Found 'class' key in algo config, mapping to '_target_' for Hydra.")
            algo_cfg["_target_"] = algo_cfg["class"]
        elif "target" in algo_cfg:
            print("[WARNING] Found 'target' key in algo config, mapping to '_target_' for Hydra.")
            algo_cfg["_target_"] = algo_cfg["target"]
        else:
            # 严重错误：无法实例化
            print(f"[ERROR] 'algo' config keys: {list(algo_cfg.keys())}")
            raise ValueError(
                "The 'algo' config in your YAML file is missing the '_target_' key. "
                "Hydra needs this to know which class to instantiate (e.g., 'algos.cql.CQL')."
            )

    algo_cfg["observation_dim"] = obs_dim
    algo_cfg["action_dim"] = act_dim
    algo_cfg["device"] = device

    # 实例化策略
    try:
        policy = hydra.utils.instantiate(algo_cfg, _recursive_=True)
    except Exception as e:
        print(f"[ERROR] Failed to instantiate algorithm: {e}")
        raise e

    # 二次检查：确保 policy 是对象且有 train 方法
    if isinstance(policy, DictConfig) or isinstance(policy, dict):
        raise TypeError(
            f"Failed to instantiate policy! Result is still a dictionary: {policy}.\n"
            "This usually means '_target_' is missing or incorrect in the config."
        )
    
    if not hasattr(policy, "train"):
        raise AttributeError(f"Instantiated policy object {type(policy)} has no method 'train()'.")

    print(f"Algorithm: {policy.__class__.__name__}")
    print(f"Device: {getattr(policy, 'device', 'unknown')}")

    update_actor_spec(policy, obs_dim, act_dim)

    print("\nStarting training...")
    total_steps = 0
    step_idx = 0

    best_eval_return = float("-inf")

    for epoch in range(num_epochs):
        # 这里的 policy.train() 之前会报错，现在经过上面的检查应该安全了
        policy.train()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        epoch_loss, step_idx = train_one_epoch(
            policy=policy,
            dataset=dataset,
            batch_size=batch_size,
            step_idx=step_idx,
            log_interval=log_interval,
            epoch=epoch,
        )

        total_steps = step_idx

        if (epoch + 1) % eval_freq == 0:
            print(f"\nEvaluating at epoch {epoch + 1}...")
            policy.eval()
            eval_metrics = evaluate(
                policy=policy,
                env_name=env_name,
                num_episodes=eval_num_episodes,
                seed=seed + epoch,
            )
            eval_return = eval_metrics.get("eval/return", 0.0)
            print(f"Evaluation return: {eval_return:.2f}")

            if eval_return > best_eval_return:
                best_eval_return = eval_return
                print(f"New best return: {best_eval_return:.2f}")

                best_checkpoint_path = output_dir / "checkpoints" / "best_model.pt"
                best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                policy.save(best_checkpoint_path)
                print(f"Saved best model to {best_checkpoint_path}")

        if (epoch + 1) % save_freq == 0:
            checkpoint_path = save_checkpoint(
                policy=policy,
                output_dir=output_dir,
                epoch=epoch + 1,
                step=step_idx,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best evaluation return: {best_eval_return:.2f}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    final_checkpoint_path = output_dir / "checkpoints" / "final_model.pt"
    policy.save(final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")


if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    config_dir = current_dir / "configs"

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    try:
        with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.1"):
            cfg = compose(config_name="experiments/d4rl_cql")
            setup_globals()
            main(cfg)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
