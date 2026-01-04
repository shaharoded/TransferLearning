from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.ffnn import PolicyNetwork, ValueNetwork
from src.adapters import EnvSpec, build_action_mask, mask_logits, map_action, pad_obs


@dataclass
class AgentConfig:
    gamma: float = 0.99
    hidden_sizes: Sequence[int] = (128,)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    grad_clip_norm: Optional[float] = None # <- Possibly better to avoid, bad for actor-critic stability


class Agent:
    """
    Base class kept intentionally lean:
    - owns device
    - owns env spec (dims, valid actions)
    - provides observation preprocessing
    """
    def __init__(self, env_spec: EnvSpec, device: Optional[torch.device] = None, seed: Optional[int] = None):
        self.env_spec = env_spec
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Pads obs to env_spec.state_dim and returns tensor [1, state_dim]
        """
        obs_pad = pad_obs(obs, self.env_spec.state_dim)
        return torch.tensor(obs_pad, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, obs: np.ndarray):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError


class ActorCriticAgent(Agent):
    """
    One-step Actor-Critic (A2C style, single environment stream):
    - Actor: categorical policy over ACTION_DIM
    - Critic: state-value V(s)
    - Mask invalid/padded actions so they are never chosen
    - Optional discretization mapping for continuous envs via EnvSpec.action_map

    Supports arbitrary-depth MLPs via hidden_sizes, e.g. [128, 128].
    """
    def __init__(self, env_spec: EnvSpec, config: Optional[AgentConfig] = None, 
                 device: Optional[torch.device] = None, seed: Optional[int] = None):
        super().__init__(env_spec=env_spec, device=device, seed=seed)
        self.cfg = config or AgentConfig()
        hidden_sizes = self.cfg.hidden_sizes

        if not isinstance(hidden_sizes, (list, tuple)) or len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must be a non-empty list/tuple, e.g. [128] or [128, 128].")

        # ffnn.py must support hidden_sizes for these constructors
        self.actor = PolicyNetwork(
            input_dim=env_spec.state_dim,
            output_dim=env_spec.action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)

        self.critic = ValueNetwork(
            input_dim=env_spec.state_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.action_mask = build_action_mask(
            action_dim=env_spec.action_dim,
            valid_indices=env_spec.valid_action_indices,
            device=self.device,
        )

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, Union[np.ndarray, int]]:
        """
        Returns:
        - action_idx: int in [0..action_dim-1]
        - env_action: int (discrete) or np.array([force]) for MountainCarContinuous
        """
        s = self.preprocess_obs(obs)          # [1, state_dim]
        logits = self.actor(s)               # [1, action_dim]
        logits = mask_logits(logits, self.action_mask)

        dist = Categorical(logits=logits)
        action_idx = int(dist.sample().item())
        env_action = map_action(self.env_spec, action_idx)
        return action_idx, env_action

    def step_update(self, obs: np.ndarray, action_idx: int, reward: float, next_obs: np.ndarray, done: bool, discount: float = 1.0) -> Dict[str, Any]:
        """
        Performs a single TD(0) update from transition (s, a, r, s', done).

        Returns a dict of scalars useful for logging/plotting:
        - actor_loss, critic_loss, entropy, td_error, v, v_next
        """
        s = self.preprocess_obs(obs)
        s_next = self.preprocess_obs(next_obs)

        logits = self.actor(s)
        logits = mask_logits(logits, self.action_mask)
        dist = Categorical(logits=logits)

        action_t = torch.tensor(action_idx, dtype=torch.int64, device=self.device)
        logp = dist.log_prob(action_t)
        ent = dist.entropy().mean()

        v = self.critic(s).squeeze(-1)

        with torch.no_grad():
            if done:
                v_next = torch.zeros_like(v)  # match old behavior :contentReference[oaicite:2]{index=2}
                td_target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            else:
                v_next = self.critic(s_next).squeeze(-1)
                td_target = torch.tensor([reward], dtype=torch.float32, device=self.device) + self.cfg.gamma * v_next

        td_error = td_target - v

        # Multiply both losses by discount (I)
        critic_loss = discount * F.mse_loss(v, td_target)
        actor_loss = -discount * (logp * td_error.detach() + self.cfg.entropy_coef * ent)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_opt.step()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(ent.item()),
            "td_error": float(td_error.mean().item()),
            "v": float(v.mean().item()),
            "v_next": float(v_next.mean().item()),
        }