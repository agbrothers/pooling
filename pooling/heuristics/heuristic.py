import numpy as np
from ray.rllib.policy.policy import Policy
from gymnasium.spaces import Discrete, Box, MultiDiscrete, Dict

class RllibHeuristic(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = args[0]
        self.action_space = args[1]
        self._weights = {}
        self.view_requirements["prev_actions"].used_for_compute_actions = False

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        env = None
        # action_list = [np.array([self._anvil_agent.act(env, obs)]) for obs in obs_batch]
        # actions = np.concatenate(action_list, axis=0)
        raise NotImplementedError
        return actions, [], {}

    def act(self, env, obs, **kwargs):
        if not self.initialized:
            self._initialize(env)
        # action, _, _ = self.policy.compute_single_action(obs)
        action = self._compute_action(obs)
        self.inputs = action
        return self.inputs

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        pass  # no update should be required here
