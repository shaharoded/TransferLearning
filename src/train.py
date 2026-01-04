from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym


def moving_average(values: list[float], window: int) -> float:
    if not values:
        return float("nan")
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def train_actor_critic(
    *,
    env: gym.Env,
    agent,
    max_episodes: int,
    max_steps_per_episode: int,
    ma_window: int,
    target_reward: Optional[float] = None,
    verbose: bool = True,
    print_every: int = 50,
    reset_seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generic trainer for one environment, using a pre-created agent and env.

    Responsibilities:
      - run episodes
      - call agent.select_action and agent.step_update
      - track metrics and convergence

    Caller responsibilities:
      - create/env wrap/seed as desired
      - create agent with correct env_spec/hyperparams/architecture
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
            print(
                f"[{env_name}] ep={ep:4d} "
                f"R={ep_reward:8.2f} "
                f"MA({ma_window})={ma:8.2f} "
                f"len={ep_len:4d}"
            )

        if target_reward is not None and len(rewards) >= ma_window and ma >= target_reward:
            converged = True
            if verbose:
                env_name = getattr(getattr(agent, "env_spec", None), "name", "env")
                print(
                    f"\n[{env_name}] Converged at episode {ep} "
                    f"(MA({ma_window})={ma:.2f} >= {target_reward:.2f}).\n"
                )
            break

    return {
        "env_name": getattr(getattr(agent, "env_spec", None), "name", None),
        "gym_id": getattr(env, "spec", None).id if getattr(env, "spec", None) is not None else None,
        "converged": converged,
        "episodes_trained": len(rewards),
        "steps_trained": steps_trained,
        "seconds_elapsed": float(time.time() - t0),
        "target_reward": target_reward,
        "ma_window": ma_window,
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "episode_lengths": ep_steps,
        "actor_loss": actor_loss_ep,
        "critic_loss": critic_loss_ep,
        "entropy": entropy_ep,
        "td_error": td_error_ep,
    }