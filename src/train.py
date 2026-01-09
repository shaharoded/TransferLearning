from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym

from src.agent import ActorCriticAgent, AgentConfig
from src.task_adapters import EnvSpec


def moving_average(values: list[float], window: int) -> float:
    if not values:
        return float("nan")
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def is_episode_successful(ep_len: int, ep_reward: float, success_criterion: Optional[Dict[str, Any]], max_steps: int) -> bool:
    """
    Determine if an episode was successful based on the criterion.
    Falls back to checking if episode finished early (len < max_steps).
    """
    if success_criterion is None:
        return ep_len < max_steps  # Default: early termination = success
    
    metric_type = success_criterion.get("type", "length")
    threshold = success_criterion.get("threshold")
    comparison = success_criterion.get("comparison", "less")
    
    if metric_type == "length":
        value = ep_len
    elif metric_type == "reward":
        value = ep_reward
    else:
        return ep_len < max_steps  # Fallback
    
    if comparison == "less":
        return value < threshold
    elif comparison == "less_equal":
        return value <= threshold
    elif comparison == "greater":
        return value > threshold
    elif comparison == "greater_equal":
        return value >= threshold
    else:
        return ep_len < max_steps  # Fallback


def train_actor_critic(
    *,
    env: gym.Env,
    agent,
    max_episodes: int,
    max_steps_per_episode: int,
    ma_window: int,
    target_success_rate: Optional[float] = None,
    success_criterion: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    print_every: int = 50,
    reset_seed_base: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic trainer for one environment, using a pre-created agent and env.

    Convergence: Stops when target_success_rate is reached (X% of recent episodes meet success_criterion)
      
    success_criterion: Dict specifying how to determine episode success.
        Examples:
        - {"type": "length", "threshold": 999, "comparison": "less"} # MountainCar: finishes early
        - {"type": "length", "threshold": 475, "comparison": "greater_equal"} # CartPole: lasts long
        - {"type": "reward", "threshold": -100, "comparison": "greater_equal"} # Acrobot: good reward
    
    If no target_success_rate provided, trains for max_episodes without early stopping.
    """
    rewards: list[float] = []
    ma_rewards: list[float] = []
    ep_steps: list[int] = []

    actor_loss_ep: list[float] = []
    critic_loss_ep: list[float] = []
    entropy_ep: list[float] = []
    td_error_ep: list[float] = []

    steps_trained = 0
    converged = False
    t0 = time.time()

    # Set default save path using environment name
    if save_path is None and hasattr(env, 'spec') and env.spec is not None:
        save_path = f"models/{env.spec.id}.pth"
    elif save_path is None:
        save_path = "models/default_model.pth"

    for ep in range(1, max_episodes + 1):
        # Optional: deterministic-but-varying resets controlled by notebook
        if reset_seed_base is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=reset_seed_base + ep)

        ep_reward = 0.0
        ep_len = 0

        actor_losses = []
        critic_losses = []
        entropies = []
        td_errors = []

        # If you're using the old "I *= gamma" weighting:
        I = 1.0

        for _ in range(max_steps_per_episode):
            action_idx, env_action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            # If you implemented discount in agent.step_update:
            try:
                logs = agent.step_update(
                    obs=obs,
                    action_idx=action_idx,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=done,
                    discount=I,
                )
            except TypeError:
                # Backward compatible if step_update doesn't accept discount
                logs = agent.step_update(
                    obs=obs,
                    action_idx=action_idx,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=done,
                )

            ep_reward += float(reward)
            ep_len += 1
            steps_trained += 1

            actor_losses.append(logs.get("actor_loss", 0.0))
            critic_losses.append(logs.get("critic_loss", 0.0))
            entropies.append(logs.get("entropy", 0.0))
            td_errors.append(logs.get("td_error", 0.0))

            obs = next_obs

            # Update I if agent has cfg.gamma, else skip
            if hasattr(agent, "cfg") and hasattr(agent.cfg, "gamma"):
                I *= float(agent.cfg.gamma)

            if done:
                break

        rewards.append(ep_reward)
        ma = moving_average(rewards, ma_window)
        ma_rewards.append(ma)
        ep_steps.append(ep_len)

        actor_loss_ep.append(float(np.mean(actor_losses)) if actor_losses else 0.0)
        critic_loss_ep.append(float(np.mean(critic_losses)) if critic_losses else 0.0)
        entropy_ep.append(float(np.mean(entropies)) if entropies else 0.0)
        td_error_ep.append(float(np.mean(td_errors)) if td_errors else 0.0)

        if verbose and (ep == 1 or ep % print_every == 0):
            env_name = getattr(getattr(agent, "env_spec", None), "name", "env")
            ma_len = moving_average(ep_steps, ma_window)
            # Calculate success rate using configurable criterion
            if len(ep_steps) >= ma_window:
                recent_steps = ep_steps[-ma_window:]
                recent_rewards = rewards[-ma_window:]
                success_rate = sum(1 for i, s in enumerate(recent_steps) 
                                  if is_episode_successful(s, recent_rewards[i], success_criterion, max_steps_per_episode)) / len(recent_steps)
            else:
                success_rate = sum(1 for i, s in enumerate(ep_steps) 
                                  if is_episode_successful(s, rewards[i], success_criterion, max_steps_per_episode)) / len(ep_steps) if ep_steps else 0.0
            
            print(
                f"[{env_name}] ep={ep:4d} "
                f"MA_R({ma_window})={ma:8.2f} "
                f"MA_len({ma_window})={ma_len:6.1f} "
                f"success_rate={success_rate:.2%}"
            )

        # Check convergence criteria
        if target_success_rate is not None and len(rewards) >= ma_window:
            # Calculate current success rate using configurable criterion
            recent_steps = ep_steps[-ma_window:]
            recent_rewards = rewards[-ma_window:]
            current_success_rate = sum(1 for i, s in enumerate(recent_steps) 
                                      if is_episode_successful(s, recent_rewards[i], success_criterion, max_steps_per_episode)) / len(recent_steps)
            
            if current_success_rate >= target_success_rate:
                converged = True
                if verbose:
                    env_name = getattr(getattr(agent, "env_spec", None), "name", "env")
                    print(
                        f"\n[{env_name}] Converged at episode {ep} "
                        f"(success_rate={current_success_rate:.2%} >= {target_success_rate:.2%})\n"
                    )
                
                # Save model if path provided
                if save_path is not None:
                    agent.save_model(save_path)
                
                break

    return {
        "env_name": getattr(getattr(agent, "env_spec", None), "name", None),
        "gym_id": getattr(env, "spec", None).id if getattr(env, "spec", None) is not None else None,
        "converged": converged,
        "episodes_trained": len(rewards),
        "steps_trained": steps_trained,
        "seconds_elapsed": float(time.time() - t0),
        "ma_window": ma_window,
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "episode_lengths": ep_steps,
        "actor_loss": actor_loss_ep,
        "critic_loss": critic_loss_ep,
        "entropy": entropy_ep,
        "td_error": td_error_ep,
    }

def finetune_actor_critic(
    *,
    source_ckpt_path: str,
    target_env: gym.Env,
    target_env_spec: EnvSpec,
    seed: int,
    max_episodes: int,
    max_steps_per_episode: int,
    ma_window: int,
    target_success_rate: Optional[float] = None,
    success_criterion: Optional[Dict[str, Any]] = None,
    agent_cfg: Optional[AgentConfig] = None,
    verbose: bool = True,
    print_every: int = 50,
    reset_seed_base: Optional[int] = None,
    save_path: Optional[str] = None,
    reinit_actor_head: bool = True,
    reinit_critic_head: bool = True,
) -> Dict[str, Any]:
    """
    Fine-tune a source-trained actor-critic on a target task
    by reusing the representation and reinitializing output layers.
    """
    # 1) Load source agent
    source_agent = ActorCriticAgent.load_model(source_ckpt_path)

    # 2) Decide configuration
    if agent_cfg is None:
        agent_cfg = source_agent.cfg

    # 3) Build fresh target agent (correct env_spec + mask)
    target_agent = ActorCriticAgent(
        env_spec=target_env_spec,
        config=agent_cfg,
        seed=seed,
    )

    # 4) Transfer trunks and reinit heads
    target_agent.load_trunks_from(source_agent)
    target_agent.reinit_output_layers(
        actor=reinit_actor_head,
        critic=reinit_critic_head,
    )

    # 5) Train using success-rate convergence
    return train_actor_critic(
        env=target_env,
        agent=target_agent,
        max_episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
        ma_window=ma_window,
        target_success_rate=target_success_rate,
        success_criterion=success_criterion,
        verbose=verbose,
        print_every=print_every,
        reset_seed_base=reset_seed_base,
        save_path=save_path,
    )