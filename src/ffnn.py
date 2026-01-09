from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence, List, Optional, Tuple

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
) -> nn.Sequential:
    """
    Builds an MLP:
      input_dim -> hidden_sizes... -> output_dim

    No activation on output layer (policy uses logits; value outputs a scalar).
    """
    if len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must have at least 1 layer, e.g. [128]")

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation.__class__())  # create a fresh activation module
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """
    Parametrized Policy Network (Actor).
    
    Architecture:
        Input: State vector (dimension: state_dim)
        Hidden: Fully connected layers with ReLU activation
        Output: Logits for each action (dimension: action_dim)
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int] = (128,)):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_sizes, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits. The Softmax is applied implicitly 
        # by the Categorical distribution in the agent.
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    State-Value Function Approximation (Baseline/Critic).
    
    Architecture:
        Input: State vector (dimension: state_dim)
        Hidden: Fully connected layers with ReLU activation
        Output: Scalar value V(s) (dimension: 1)
    """
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (128,)):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_sizes, 1) # Output is a single scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class LateralAdapter(nn.Module):
    """
    Lateral connection adapter for Progressive Neural Networks.
    
    It projects source features into a compatible space for the target network,
    applying non-linearity (ReLU). This allows the network to:
    1. Compress dimensions (solving parameter explosion).
    2. Filter out irrelevant/harmful source features (solving negative transfer).
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

        # Initialize weights to be very small. 
        # This ensures the source features start as "silent" and don't 
        # interfere with the target's initial learning.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1e-4) # Start almost at zero
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProgressiveNetwork(nn.Module):
    """
    A generic Progressive Neural Network module that implements the fusion logic.
    
    This module represents a single "head" (either Actor or Critic) of a Progressive Network.
    It orchestrates the forward pass by:
    1. Processing the input through the Target's own trunk (hidden layers).
    2. Processing the input through all frozen Source trunks in parallel.
    3. Concatenating (fusing) the features from all trunks.
    4. Passing the fused representation to a specific output head.

    It handles variable input dimensions by zero-padding the observations to match
    the expected input size of each specific trunk.

    Architecture:
        Input -> [Pad] -> Target Trunk -> Features_T
        Input -> [Pad] -> Source Trunk 1 -> Features_S1 (Frozen)
        Input -> [Pad] -> Source Trunk 2 -> Features_S2 (Frozen)
              ...
        [Features_T, Features_S1, Features_S2] -> Concat -> Output Head -> Result
    """
    def __init__(
        self, 
        target_trunk: nn.Sequential, 
        source_trunks: nn.ModuleList, 
        adapters: nn.ModuleList,
        head: nn.Linear, 
        target_dim: int, 
        source_dims: List[int]
    ):
        """
        Args:
            target_trunk: The hidden layers of the target network (trainable).
            source_trunks: A list of hidden layers from source networks (frozen).
            adapters: A list of adapter modules applied to source features.
            head: The final output layer (Linear) accepting the fused dimension.
            target_dim: The expected input state dimension for the target trunk.
            source_dims: A list of expected input state dimensions for each source trunk.
            norm: LayerNorm applied to fused features before head.
        """
        super().__init__()
        self.target_trunk = target_trunk
        self.source_trunks = source_trunks
        self.adapters = adapters
        self.head = head
        self.target_dim = target_dim
        self.source_dims = source_dims

        # Calculate the size of the fused vector (Input to the head)
        self.fused_dim = head.in_features 
        self.norm = nn.LayerNorm(self.fused_dim)

    def _pad_input(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Zero-pads the input tensor to match the specific trunk's required dimension.
        
        This enables transfer learning between environments with different state spaces
        (e.g., CartPole (4) -> Acrobot (6)).

        Args:
            x: Input tensor of shape [batch, current_dim]
            target_dim: The required dimension for the specific trunk.
            
        Returns:
            Tensor of shape [batch, target_dim]
        """
        current_dim = x.shape[-1]
        if current_dim >= target_dim:
            return x[..., :target_dim]
        batch_shape = x.shape[:-1]
        padding = torch.zeros(*batch_shape, target_dim - current_dim, 
                             dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass with lateral connections.
        Returns:
            Output of the head (logits for Actor, scalar value for Critic).
        """
        # 1. Target Path
        x_target = self._pad_input(x, self.target_dim)
        target_feat = self.target_trunk(x_target)
        
        # 2. Source Paths (Frozen + Adapted)
        source_feats = []
        for trunk, adapter, dim in zip(self.source_trunks, self.adapters, self.source_dims):
            x_src = self._pad_input(x, dim)
            with torch.no_grad():
                raw_feat = trunk(x_src) # Extract features
            
            # Apply Adapter (This is the new part!)
            # We allow gradients here so the adapter can learn what to pass through
            adapted_feat = adapter(raw_feat) 
            source_feats.append(adapted_feat)
        
        # 3. Fusion & Head
        fused = torch.cat([target_feat] + source_feats, dim=-1)
        fused = self.norm(fused)
        return self.head(fused)


class ProgressiveWrapper(nn.Module):
    """
    A Factory/Container that builds a complete Progressive Actor-Critic architecture.

    This class acts as the bridge between the high-level Agents and the low-level 
    Neural Networks. It performs the "surgery" required to construct a Progressive Network:
    
    1.  **Extraction:** It takes fully constructed Agent instances (Source and Target) 
        and extracts their internal neural networks.
    2.  **Slicing:** It separates the "trunk" (hidden layers) from the "head" (output layer).
        We assume the last layer of the network is the head.
    3.  **Freezing:** It explicitly freezes all parameters of the source agents to 
        prevent catastrophic forgetting and ensure we only learn the new task.
    4.  **Construction:** It initializes two `ProgressiveNetwork` instances:
        one for the Actor and one for the Critic.

    This design allows the `ProgressiveAgent` to treat `.actor` and `.critic` as 
    standard PyTorch modules, oblivious to the complexity of the underlying fusion.
    """
    def __init__(self, target_agent, source_agents):
        """
        Initialize the wrapper and build the progressive networks.

        Args:
            target_agent: The generic ActorCriticAgent to be trained (must have .actor and .critic).
            source_agents: List of pre-trained ActorCriticAgents (will be frozen).
        """
        super().__init__()
        
        # 1. Setup Configuration
        self.device = target_agent.device
        target_state_dim = target_agent.env_spec.state_dim
        source_state_dims = [s.env_spec.state_dim for s in source_agents]
        
        target_hidden_dim = target_agent.cfg.hidden_sizes[-1]
        
        # ADAPTER CONFIGURATION
        # We project all sources to match the target's hidden dimension.
        # This keeps the fusion balanced (Target vs Source1 vs Source2)
        adapter_output_dim = target_hidden_dim // 4  # Compress to 1/4th of target hidden size
        
        # New fused dim is: Target + (Adapter_Size * Num_Sources)
        fused_dim = target_hidden_dim + (adapter_output_dim * len(source_agents))

        print(f"✓ Initializing Progressive Wrapper with Adapters")
        print(f"  Fused Dim: {fused_dim} (Target: {target_hidden_dim} + {len(source_agents)}x Adapters: {adapter_output_dim})")

        # 2. Extract Trunks
        target_actor_trunk = nn.Sequential(*list(target_agent.actor.net)[:-1])
        target_critic_trunk = nn.Sequential(*list(target_agent.critic.net)[:-1])

        source_actor_trunks = nn.ModuleList()
        source_critic_trunks = nn.ModuleList()
        
        # 3. CREATE ADAPTERS
        actor_adapters = nn.ModuleList()
        critic_adapters = nn.ModuleList()

        for i, src in enumerate(source_agents):
            # Freeze sources
            for param in src.actor.parameters(): param.requires_grad = False
            for param in src.critic.parameters(): param.requires_grad = False
            
            # Extract trunks
            src_actor_hidden = src.cfg.hidden_sizes[-1]
            src_critic_hidden = src.cfg.hidden_sizes[-1] # Usually same
            
            source_actor_trunks.append(nn.Sequential(*list(src.actor.net)[:-1]))
            source_critic_trunks.append(nn.Sequential(*list(src.critic.net)[:-1]))
            
            # Add Adapters: Source Hidden -> Target Hidden
            actor_adapters.append(LateralAdapter(src_actor_hidden, adapter_output_dim))
            critic_adapters.append(LateralAdapter(src_critic_hidden, adapter_output_dim))

        # 4. Create New Output Heads
        new_actor_head = nn.Linear(fused_dim, target_agent.env_spec.action_dim)
        new_critic_head = nn.Linear(fused_dim, 1)

        # Init heads
        nn.init.orthogonal_(new_actor_head.weight, gain=0.01)
        nn.init.zeros_(new_actor_head.bias)
        nn.init.orthogonal_(new_critic_head.weight, gain=1.0)
        nn.init.zeros_(new_critic_head.bias)

        # 5. Construct Networks
        self.actor = ProgressiveNetwork(
            target_trunk=target_actor_trunk,
            source_trunks=source_actor_trunks,
            adapters=actor_adapters,     # <--- Pass adapters
            head=new_actor_head,
            target_dim=target_state_dim,
            source_dims=source_state_dims
        )

        self.critic = ProgressiveNetwork(
            target_trunk=target_critic_trunk,
            source_trunks=source_critic_trunks,
            adapters=critic_adapters,
            head=new_critic_head,
            target_dim=target_state_dim,
            source_dims=source_state_dims
        )