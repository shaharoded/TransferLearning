from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence

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