from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def summarize_run(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact summary suitable for printing / tabulating in the notebook.
    """
    ma = history.get("ma_rewards", [])
    final_ma = float(ma[-1]) if ma else None

    return {
        "env": history.get("env_name"),
        "gym_id": history.get("gym_id"),
        "episodes": int(history.get("episodes_trained", 0)),
        "steps": int(history.get("steps_trained", 0)),
        "seconds": round(float(history.get("seconds_elapsed", 0.0)), 2),
        "converged": bool(history.get("converged", False)),
        "final_ma": round(final_ma, 3) if final_ma is not None else None,
        "target": history.get("target_reward"),
        "ma_window": history.get("ma_window"),
    }


def plot_training_curve_single(
    history: Dict[str, Any],
    *,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
    show_raw: bool = True,
    show_ma: bool = True,
) -> None:
    """
    Single-env plot (one figure, one axes).
    """
    rewards = history["rewards"]
    ma_rewards = history.get("ma_rewards", None)
    ma_window = history.get("ma_window", None)

    x = np.arange(1, len(rewards) + 1)

    plt.figure()
    if show_raw:
        plt.plot(x, rewards, label="Episode reward")
    if show_ma and ma_rewards is not None:
        label = f"Moving avg ({ma_window})" if ma_window is not None else "Moving avg"
        plt.plot(x, ma_rewards, label=label)

    if threshold is None:
        threshold = history.get("target_reward", None)
    if threshold is not None:
        plt.axhline(y=float(threshold), linestyle="--", label="Target threshold")

    if title is None:
        title = f"{history.get('env_name', 'env')} training"
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_diagnostics_single(
    history: Dict[str, Any],
    *,
    title: Optional[str] = None,
    keys: Sequence[str] = ("actor_loss", "critic_loss", "entropy", "td_error"),
) -> None:
    """
    Single-env diagnostics figure.
    """
    x = np.arange(1, history["episodes_trained"] + 1)

    plt.figure()
    for k in keys:
        if k in history:
            plt.plot(x, history[k], label=k)

    if title is None:
        title = f"{history.get('env_name', 'env')} diagnostics"
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_training_curves_grid(
    histories: Dict[str, Dict[str, Any]],
    *,
    order: Optional[Sequence[str]] = None,
    titles: Optional[Dict[str, str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    show_raw: bool = True,
    show_ma: bool = True,
    grid_shape: Tuple[int, int] = (1, 3),
    figsize: Tuple[int, int] = (18, 4),
) -> None:
    """
    Plots multiple envs in ONE figure with multiple panels (subplots).
    Default is a 1x3 grid: CartPole, Acrobot, MountainCar (or whatever you pass).

    histories: dict env_name -> history dict returned from train_actor_critic
    thresholds: dict env_name -> float target threshold (optional). If not provided,
                will use history['target_reward'] when available.
    """
    if order is None:
        order = list(histories.keys())

    n = len(order)
    rows, cols = grid_shape
    if rows * cols < n:
        raise ValueError(f"grid_shape {grid_shape} too small for {n} plots")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for i, env_name in enumerate(order):
        h = histories[env_name]
        ax = axes[i]

        rewards = h["rewards"]
        ma_rewards = h.get("ma_rewards", None)
        ma_window = h.get("ma_window", None)
        x = np.arange(1, len(rewards) + 1)

        if show_raw:
            ax.plot(x, rewards, label="Reward")
        if show_ma and ma_rewards is not None:
            label = f"MA({ma_window})" if ma_window is not None else "MA"
            ax.plot(x, ma_rewards, label=label)

        thr = None
        if thresholds and env_name in thresholds:
            thr = thresholds[env_name]
        else:
            thr = h.get("target_reward", None)

        if thr is not None:
            ax.axhline(y=float(thr), linestyle="--", label="Target")

        t = titles[env_name] if titles and env_name in titles else env_name
        ax.set_title(t)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)
        ax.legend()

    # Hide any unused axes
    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_diagnostics_grid(
    histories: Dict[str, Dict[str, Any]],
    *,
    order: Optional[Sequence[str]] = None,
    keys: Sequence[str] = ("actor_loss", "critic_loss", "entropy", "td_error"),
    grid_shape: Tuple[int, int] = (1, 3),
    figsize: Tuple[int, int] = (18, 4),
) -> None:
    """
    ONE figure with multiple panels: diagnostics per env.
    """
    if order is None:
        order = list(histories.keys())

    n = len(order)
    rows, cols = grid_shape
    if rows * cols < n:
        raise ValueError(f"grid_shape {grid_shape} too small for {n} plots")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, env_name in enumerate(order):
        h = histories[env_name]
        ax = axes[i]
        x = np.arange(1, h["episodes_trained"] + 1)

        for k in keys:
            if k in h:
                ax.plot(x, h[k], label=k)

        ax.set_title(f"{env_name} diagnostics")
        ax.set_xlabel("Episode")
        ax.grid(True)
        ax.legend()

    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()