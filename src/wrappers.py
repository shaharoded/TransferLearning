"""
Gym environment wrappers for reward shaping and other modifications.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


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
