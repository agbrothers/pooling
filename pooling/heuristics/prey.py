import numpy as np
from ray.rllib.policy.policy import Policy


class PreyHeuristic(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = args[0]
        self.action_space = args[1]
        self._weights = {}
        self.view_requirements["prev_actions"].used_for_compute_actions = False
        # self.prey_type = 3
        self.pred_type = 2
        self.seed = args[2]["seed"]

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
        """
        
        Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`
                                                0          1          2           3         4

        """

        ## SLICE CLOSEST (DISTANCE) PREY ENTITY FOR EACH ITEM IN BATCH
        ## obs = [batch, entities, features]
        pred_mask = obs[:, 1:, -1] != self.pred_type
        assert not np.all(pred_mask.min(axis=1)), "No pred entities in observation, check your env_config!"
        pred_obs = obs[:, 1:, :4] * (pred_mask*1e+6+1)[..., None] #.reshape(len(obs), -1)
        pred_pos = pred_obs[..., :2] + pred_obs[..., 2:4]
        pred_dist = np.linalg.norm(pred_pos, axis=-1, keepdims=True)
        pred_closest = pred_dist == np.min(pred_dist, axis=1)[...,None]
        
        ## COMPUTE THE DIRECTION IN WHICH TO MOVE
        pred_target = pred_pos[pred_closest[..., 0]]
        direction = np.max(np.abs(pred_target), axis=1, keepdims=True) == np.abs(pred_target)
        sign = np.sign(pred_target[direction])
        ## CONVERT DIRECTION AND SIGN VALUES INTO INTEGER ACTIONS
        actions = (np.argmax(direction, axis=1) * 2 + 2) - (sign + 1)//2
        return actions.astype(int), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        pass  # no update should be required here
