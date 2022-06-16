"""Common configuration element for scripts learning from demonstrations."""

import logging
from typing import Optional, Sequence

import gym
import sacred
from stable_baselines3.common import policies
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper

demonstrations_ingredient = sacred.Ingredient("demonstrations")
logger = logging.getLogger(__name__)


@demonstrations_ingredient.config
def config():
    # Demonstrations
    n_expert_demos = 50  # Num demos used.
    locals()  # quieten flake8


@demonstrations_ingredient.named_config
def fast():
    n_expert_demos = 1  # noqa: F841


@demonstrations_ingredient.capture
def generate_expert_trajs(
    expert: policies.BasePolicy,
    env_name: str,
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Generates expert demonstrations.

    Args:
        expert: The expert to sample trajectories from.
        env_name: The name of the environment, that the expert was created for.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    rollout_env = DummyVecEnv(
        [lambda: RolloutInfoWrapper(gym.make(env_name)) for _ in range(4)]
    )
    return rollout.rollout(
        expert,
        rollout_env,
        rollout.make_sample_until(min_timesteps=2000, min_episodes=n_expert_demos),
    )
