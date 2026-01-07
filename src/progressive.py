"""
Progressive Networks for Transfer Learning in Actor-Critic.

This module implements the Progressive Neural Network architecture for transferring
knowledge from multiple pre-trained source models to a new target task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.adapters import EnvSpec, build_action_mask, mask_logits, map_action, pad_obs
from src.agent import ActorCriticAgent, AgentConfig


class ProgressiveActorCritic(nn.Module):
    """
    Progressive Neural Network wrapper for Actor-Critic transfer learning.
    
    This class implements the "Single Hidden Layer" connection strategy:
    1. Freeze all source models' parameters (no gradient updates)
    2. Pass input through all source models' hidden layers to extract features
    3. Pass input through target model's hidden layer
    4. Concatenate all hidden features: [target_hidden, source1_hidden, source2_hidden, ...]
    5. Feed combined features through target model's output heads (actor & critic)
    
    The target model's output layers are resized to accept the larger concatenated input.
    
    Handles different input dimensions by zero-padding smaller inputs.
    """
    
    def __init__(
        self,
        target_agent: ActorCriticAgent,
        source_agents: List[ActorCriticAgent],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Progressive Actor-Critic network.
        
        Args:
            target_agent: Fresh agent for the target task (will be modified)
            source_agents: List of pre-trained source agents (will be frozen)
            device: Device to place the models on
        """
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_agent = target_agent
        self.source_agents = source_agents
        self.env_spec = target_agent.env_spec
        self.cfg = target_agent.cfg
        
        # Freeze all source models
        for i, source in enumerate(self.source_agents):
            for param in source.actor.parameters():
                param.requires_grad = False
            for param in source.critic.parameters():
                param.requires_grad = False
            print(f"✓ Source model {i+1} ({source.env_spec.name}) frozen")
        
        # Store hidden sizes from config
        self.hidden_sizes = list(self.cfg.hidden_sizes)
        
        # Extract hidden layer components from source and target
        self._setup_progressive_architecture()
        
        # Build action mask for target environment
        self.action_mask = build_action_mask(
            action_dim=self.env_spec.action_dim,
            valid_indices=self.env_spec.valid_action_indices,
            device=self.device,
        )
        
        # Move everything to device
        self.to(self.device)
        
        print(f"✓ Progressive network initialized with {len(source_agents)} source(s)")
        print(f"  Target hidden dim: {self.target_hidden_dim}")
        print(f"  Total fused dim: {self.fused_dim}")
    
    def _setup_progressive_architecture(self):
        """
        Set up the progressive network architecture by:
        1. Extracting hidden layers from source and target
        2. Creating new output heads that accept the fused features
        """
        # Get hidden dimension (assuming single hidden layer as per architecture)
        self.target_hidden_dim = self.hidden_sizes[-1]
        
        # Each source has the same hidden dimension (shared architecture)
        self.source_hidden_dims = [self.target_hidden_dim for _ in self.source_agents]
        
        # Total fused dimension = target + all sources
        self.fused_dim = self.target_hidden_dim + sum(self.source_hidden_dims)
        
        # Extract trunk (all layers except output) from target actor and critic
        # For a network like: Linear(6,128) -> ReLU -> Linear(128,5)
        # Trunk is: Linear(6,128) -> ReLU
        self.target_actor_trunk = nn.Sequential(*list(self.target_agent.actor.net)[:-1])
        self.target_critic_trunk = nn.Sequential(*list(self.target_agent.critic.net)[:-1])
        
        # Create source trunks (frozen)
        self.source_actor_trunks = nn.ModuleList([
            nn.Sequential(*list(src.actor.net)[:-1])
            for src in self.source_agents
        ])
        self.source_critic_trunks = nn.ModuleList([
            nn.Sequential(*list(src.critic.net)[:-1])
            for src in self.source_agents
        ])
        
        # Create new output heads that take the fused features
        self.actor_head = nn.Linear(self.fused_dim, self.env_spec.action_dim)
        self.critic_head = nn.Linear(self.fused_dim, 1)
        
        # Initialize heads with appropriate gains
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)
        
        # Store source state dimensions for padding
        self.source_state_dims = [src.env_spec.state_dim for src in self.source_agents]
        self.target_state_dim = self.target_agent.env_spec.state_dim
        self.max_state_dim = max([self.target_state_dim] + self.source_state_dims)
    
    def _pad_input(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Pad input tensor with zeros to match target dimension.
        
        Args:
            x: Input tensor of shape [batch, current_dim]
            target_dim: Target dimension to pad to
            
        Returns:
            Padded tensor of shape [batch, target_dim]
        """
        current_dim = x.shape[-1]
        if current_dim >= target_dim:
            return x[..., :target_dim]
        
        # Create zero padding
        batch_shape = x.shape[:-1]
        padding = torch.zeros(*batch_shape, target_dim - current_dim, 
                             dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=-1)
    
    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the actor (policy) network.
        
        Args:
            x: State tensor of shape [batch, state_dim]
            
        Returns:
            Action logits of shape [batch, action_dim]
        """
        # Get target hidden features
        x_target = self._pad_input(x, self.target_state_dim)
        target_features = self.target_actor_trunk(x_target)
        
        # Get source hidden features (with zero-padding if needed)
        source_features = []
        for i, (trunk, src_dim) in enumerate(zip(self.source_actor_trunks, self.source_state_dims)):
            x_src = self._pad_input(x, src_dim)
            with torch.no_grad():  # Ensure no gradients for frozen sources
                src_feat = trunk(x_src)
            source_features.append(src_feat)
        
        # Concatenate: [target, source1, source2, ...]
        fused = torch.cat([target_features] + source_features, dim=-1)
        
        # Output through new head
        logits = self.actor_head(fused)
        return logits
    
    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the critic (value) network.
        
        Args:
            x: State tensor of shape [batch, state_dim]
            
        Returns:
            State value of shape [batch, 1]
        """
        # Get target hidden features
        x_target = self._pad_input(x, self.target_state_dim)
        target_features = self.target_critic_trunk(x_target)
        
        # Get source hidden features (with zero-padding if needed)
        source_features = []
        for i, (trunk, src_dim) in enumerate(zip(self.source_critic_trunks, self.source_state_dims)):
            x_src = self._pad_input(x, src_dim)
            with torch.no_grad():  # Ensure no gradients for frozen sources
                src_feat = trunk(x_src)
            source_features.append(src_feat)
        
        # Concatenate: [target, source1, source2, ...]
        fused = torch.cat([target_features] + source_features, dim=-1)
        
        # Output through new head
        value = self.critic_head(fused)
        return value
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass returning both actor logits and critic value.
        
        Args:
            x: State tensor of shape [batch, state_dim]
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        return self.forward_actor(x), self.forward_critic(x)


class ProgressiveActorCriticAgent:
    """
    Agent wrapper for Progressive Actor-Critic that provides the same interface
    as ActorCriticAgent for compatibility with the training loop.
    """
    
    def __init__(
        self,
        env_spec: EnvSpec,
        source_agents: List[ActorCriticAgent],
        config: Optional[AgentConfig] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Progressive Actor-Critic Agent.
        
        Args:
            env_spec: Target environment specification
            source_agents: List of pre-trained source agents
            config: Agent configuration
            device: Device to use
            seed: Random seed
        """
        self.env_spec = env_spec
        self.cfg = config or AgentConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        print(f"✓ Progressive Agent device: {self.device}")
        
        # Create a fresh target agent
        self.target_agent = ActorCriticAgent(
            env_spec=env_spec,
            config=self.cfg,
            device=self.device,
            seed=seed,
        )
        
        # Create the progressive network
        self.progressive_net = ProgressiveActorCritic(
            target_agent=self.target_agent,
            source_agents=source_agents,
            device=self.device,
        )
        
        # Build action mask
        self.action_mask = build_action_mask(
            action_dim=env_spec.action_dim,
            valid_indices=env_spec.valid_action_indices,
            device=self.device,
        )
        
        # Set up optimizers for trainable parameters only
        trainable_params = [p for p in self.progressive_net.parameters() if p.requires_grad]
        
        # Split params: actor (trunk + head) and critic (trunk + head)
        actor_params = (
            list(self.progressive_net.target_actor_trunk.parameters()) +
            list(self.progressive_net.actor_head.parameters())
        )
        critic_params = (
            list(self.progressive_net.target_critic_trunk.parameters()) +
            list(self.progressive_net.critic_head.parameters())
        )
        
        self.actor_opt = torch.optim.Adam(actor_params, lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=self.cfg.critic_lr)
        
        print(f"✓ Progressive Agent initialized")
        print(f"  Trainable actor params: {sum(p.numel() for p in actor_params if p.requires_grad)}")
        print(f"  Trainable critic params: {sum(p.numel() for p in critic_params if p.requires_grad)}")
    
    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Pads obs to env_spec.state_dim and returns tensor [1, state_dim].
        """
        obs_pad = pad_obs(obs, self.env_spec.state_dim)
        return torch.tensor(obs_pad, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, Union[np.ndarray, int]]:
        """
        Select action using the progressive policy.
        
        Returns:
            Tuple of (action_idx, env_action)
        """
        s = self.preprocess_obs(obs)
        logits = self.progressive_net.forward_actor(s)
        logits = mask_logits(logits, self.action_mask)
        
        dist = Categorical(logits=logits)
        action_idx = int(dist.sample().item())
        env_action = map_action(self.env_spec, action_idx)
        return action_idx, env_action
    
    def step_update(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        discount: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform a single TD(0) update from transition (s, a, r, s', done).
        
        Returns:
            Dict of scalars for logging
        """
        s = self.preprocess_obs(obs)
        s_next = self.preprocess_obs(next_obs)
        
        # Forward pass
        logits = self.progressive_net.forward_actor(s)
        logits = mask_logits(logits, self.action_mask)
        dist = Categorical(logits=logits)
        
        action_t = torch.tensor(action_idx, dtype=torch.int64, device=self.device)
        logp = dist.log_prob(action_t)
        ent = dist.entropy().mean()
        
        v = self.progressive_net.forward_critic(s).squeeze(-1)
        
        with torch.no_grad():
            if done:
                v_next = torch.zeros_like(v)
                td_target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            else:
                v_next = self.progressive_net.forward_critic(s_next).squeeze(-1)
                td_target = torch.tensor([reward], dtype=torch.float32, device=self.device) + self.cfg.gamma * v_next
        
        td_error = td_target - v
        
        # Compute losses with discount
        critic_loss = discount * F.mse_loss(v, td_target)
        actor_loss = -discount * (logp * td_error.detach() + self.cfg.entropy_coef * ent)
        
        # Update critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.progressive_net.target_critic_trunk.parameters()) +
                list(self.progressive_net.critic_head.parameters()),
                self.cfg.grad_clip_norm
            )
        self.critic_opt.step()
        
        # Update actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.progressive_net.target_actor_trunk.parameters()) +
                list(self.progressive_net.actor_head.parameters()),
                self.cfg.grad_clip_norm
            )
        self.actor_opt.step()
        
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(ent.item()),
            "td_error": float(td_error.mean().item()),
            "v": float(v.mean().item()),
            "v_next": float(v_next.mean().item()),
        }
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save the progressive network to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "progressive_net_state_dict": self.progressive_net.state_dict(),
            "env_spec": {
                "name": self.env_spec.name,
                "obs_dim": self.env_spec.obs_dim,
                "state_dim": self.env_spec.state_dim,
                "action_dim": self.env_spec.action_dim,
                "valid_action_indices": self.env_spec.valid_action_indices,
                "action_map": self.env_spec.action_map,
            },
            "config": {
                "gamma": self.cfg.gamma,
                "hidden_sizes": list(self.cfg.hidden_sizes),
                "actor_lr": self.cfg.actor_lr,
                "critic_lr": self.cfg.critic_lr,
                "entropy_coef": self.cfg.entropy_coef,
                "grad_clip_norm": self.cfg.grad_clip_norm,
            },
        }
        
        torch.save(checkpoint, path)
        print(f"✓ Progressive model saved to {path}")
