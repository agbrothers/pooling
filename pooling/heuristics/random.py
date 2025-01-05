import numpy as np
from ray.rllib.policy.policy import Policy


class RandomHeuristic(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = args[0]
        self.action_space = args[1]
        self._weights = {}
        self.view_requirements["prev_actions"].used_for_compute_actions = False
        self.seed = args[2]["seed"]
        self.action_space.seed(self.seed)

    def compute_actions(
        self,
        obs:np.ndarray,
        state_batches:list,
        prev_action_batch:np.ndarray=None,
        prev_reward_batch:np.ndarray=None,
        info_batch:dict=None,
        explore:bool=False,
        timestep:int=None,
        episodes:list=None,
        **kwargs,
    ):
        return (
            [self.action_space.sample() for _ in range(len(obs))],
            [],
            {},
        )

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        pass  # no update should be required here






