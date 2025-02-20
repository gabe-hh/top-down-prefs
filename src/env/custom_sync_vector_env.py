from __future__ import annotations

import copy
from typing import Any, Callable, Iterator, Sequence

import numpy as np

import gymnasium as gym
from gymnasium import Env, Space
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector.utils import (
    batch_differing_spaces,
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)
from gymnasium.envs.registration import _find_spec
from gymnasium.vector.vector_env import ArrayType, VectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv

class CustomSyncVectorEnv(SyncVectorEnv):
    def __init__(self, env_fns, copy=True, observation_mode="same", wrappers=None):
        super().__init__(env_fns, copy=copy, observation_mode=observation_mode)
        self.wrappers = wrappers  # store the wrappers list

    def step(self, actions):
        """Overrides step so that auto-resetting is disabled."""
        actions = iterate(self.action_space, actions)
        observations, infos = [], {}
        rewards = np.zeros((self.num_envs,), dtype=np.float64)
        terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        
        for i, action in enumerate(actions):
            # Don't autoreset env
            (
                env_obs,
                rewards[i],
                terminations[i],
                truncations[i],
                env_info,
            ) = self.envs[i].step(action)
            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)
        
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )
        # Do not change _autoreset_envs so they won't trigger resets.
        return (
            copy.deepcopy(self._observations) if self.copy else self._observations,
            np.copy(rewards),
            np.copy(terminations),
            np.copy(truncations),
            infos,
        )

    def clone_env_states(self):
        """Returns copies of each individual environment's state.
           You will need to implement `clone_env` for your MiniGrid env."""
        return [clone_env(env) for env in self.envs]
    
    def clone(self):
        """
        Creates a new vectorized environment (of the same type) with cloned states
        from each individual environment.
        """
        # Clone each individual environment.
        cloned_envs = [clone_env(env, wrappers=self.wrappers) for env in self.envs]
        
        env_fns = [lambda cloned=cloned: cloned for cloned in cloned_envs]

        new_vec_env = CustomSyncVectorEnv(
            env_fns=env_fns,
            copy=self.copy,
            observation_mode=self.observation_mode
        )
        
        new_vec_env.unwrapped.spec = copy.deepcopy(self.unwrapped.spec)
        return new_vec_env

def clone_env(env, wrappers=None):
    new_env = gym.make(env.spec.id)

    new_env.reset()

    if wrappers is not None:
        for wrapper in wrappers:
            new_env = wrapper(new_env)

    # seems to be enough to copy the env
    new_env.unwrapped.grid = copy.deepcopy(env.unwrapped.grid)
    new_env.unwrapped.agent_pos = env.unwrapped.agent_pos
    new_env.unwrapped.agent_dir = env.unwrapped.agent_dir

    return new_env

def make_custom_vec(
    env_id: str,
    num_envs: int = 1,
    wrappers: Sequence[Callable[[Env], Env]] = [],
) -> CustomSyncVectorEnv:
    """Creates a vectorized environment.

    Args:
        env_id: The environment ID.
        num_envs: The number of environments to vectorize.
        vectorization_mode: The vectorization mode.
        wrappers: A list of wrapper functions to apply to each environment.

    Returns:
        A vectorized environment.
    """

    if wrappers is None:
        wrappers = []

    env_spec = _find_spec(env_id)

    env_spec = copy.deepcopy(env_spec)
    env_spec_kwargs = env_spec.kwargs
    env_spec.kwargs = dict()

    def create_single_env():
        env = gym.make(env_spec)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    
    env = CustomSyncVectorEnv(env_fns=[create_single_env for _ in range(num_envs)], wrappers=wrappers)

    copied_id_spec = copy.deepcopy(env_spec)
    copied_id_spec.kwargs = env_spec_kwargs
    if num_envs != 1:
        copied_id_spec.kwargs["num_envs"] = num_envs
    if len(wrappers) > 0:
        copied_id_spec.kwargs["wrappers"] = wrappers
    env.unwrapped.spec = copied_id_spec

    return env