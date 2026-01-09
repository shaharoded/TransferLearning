from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import gymnasium as gym


@dataclass(frozen=True)
class EnvSpec:
    """Contains environment specifications needed for adapters between environments."""
    name: str
    obs_dim: int
    state_dim: int
    action_dim: int
    valid_action_indices: Tuple[int, ...]
    action_map: Optional[Tuple[float, ...]] = None  # for continuous env discretization


def pad_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pads a 1D observation vector with zeros to target_dim.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[0] > target_dim:
        raise ValueError(f"obs dim {obs.shape[0]} > target_dim {target_dim}")
    if obs.shape[0] == target_dim:
        return obs
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: obs.shape[0]] = obs
    return out


def build_action_mask(action_dim: int, valid_indices: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    Returns a float mask of shape [action_dim], with 1.0 for valid actions and 0.0 for invalid.
    """
    mask = torch.zeros((action_dim,), dtype=torch.float32, device=device)
    for i in valid_indices:
        mask[i] = 1.0
    return mask


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor, invalid_fill: float = -1e9) -> torch.Tensor:
    """
    logits: [..., action_dim]
    action_mask: [action_dim] with 1 for valid, 0 for invalid

    Will push the logits of invalid actions to invalid_fill, which is very small.
    """
    if logits.shape[-1] != action_mask.shape[0]:
        raise ValueError(f"logits last dim {logits.shape[-1]} != action_mask dim {action_mask.shape[0]}")
    # broadcast mask to logits shape
    masked = logits.masked_fill(action_mask == 0, invalid_fill)
    return masked


def map_action(env_spec: EnvSpec, action_idx: int) -> np.ndarray | int:
    """
    Converts an action index in [0..action_dim-1] to the environment action.
    - Discrete envs: return int
    - MountainCarContinuous: return np.array([force], dtype=np.float32)
    """
    if env_spec.action_map is None:
        # discrete
        return int(action_idx)

    # continuous with discretization
    force = env_spec.action_map[action_idx]
    return np.array([force], dtype=np.float32)


def get_default_env_specs(state_dim: int = 6, action_dim: int = 5) -> Dict[str, EnvSpec]:
    """
    Shared state_dim=6 and action_dim=5 works for:
    - CartPole: obs_dim=4, valid actions: 0,1 (indices 2..4 are padding)
    - Acrobot: obs_dim=6, valid actions: 0,1,2 (indices 3..4 are padding)
    - MountainCarContinuous: obs_dim=2, discretized into 5 bins mapped to [-1,-0.5,0,0.5,1]
    """
    return {
        "cartpole": EnvSpec(
            name="cartpole",
            obs_dim=4,
            state_dim=state_dim,
            action_dim=action_dim,
            valid_action_indices=(0, 1),
            action_map=None,
        ),
        "acrobot": EnvSpec(
            name="acrobot",
            obs_dim=6,
            state_dim=state_dim,
            action_dim=action_dim,
            valid_action_indices=(0, 1, 2),
            action_map=None,
        ),
        "mountaincar": EnvSpec(
            name="mountaincar",
            obs_dim=2,
            state_dim=state_dim,
            action_dim=action_dim,
            valid_action_indices=(0, 1, 2, 3, 4),
            action_map=(-1.0, -0.5, 0.0, 0.5, 1.0),
        ),
    }

class MountainCarRewardShaping(gym.Wrapper):
    """
    Reward shaping for MountainCarContinuous-v0.
    
    Original reward: -0.1 * action^2 per step until goal reached (position >= 0.45)
    Shaped reward: Add bonus for:
      - Reaching higher positions (height reward)
      - Building velocity in the right direction
    
    This helps the agent learn even when it never reaches the goal initially.
    """
    def __init__(self, env: gym.Env, position_weight: float = 1.0, velocity_weight: float = 0.1):
        super().__init__(env)
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.prev_position = None
        self.best_position = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_position = float(obs[0])
        self.best_position = float(obs[0])
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        position = float(obs[0])
        velocity = float(obs[1])
        
        # Original reward is small and negative (action penalty)
        # Add shaping bonuses:
        
        # 1. Reward for reaching new heights (position progress)
        if position > self.best_position:
            height_bonus = self.position_weight * (position - self.best_position)
            reward += height_bonus
            self.best_position = position
        
        # 2. Reward for velocity in the right direction (towards the goal at 0.45)
        # When on the left side, positive velocity is good
        # When on the right side near goal, positive velocity is still good
        velocity_bonus = self.velocity_weight * abs(velocity)
        reward += velocity_bonus
        
        # 3. Large bonus for reaching the goal
        if terminated:
            reward += 100.0  # Big bonus for success
        
        self.prev_position = position
        
        return obs, reward, terminated, truncated, info
