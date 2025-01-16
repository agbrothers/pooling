import numpy as np
import gymnasium as gym
from ray.rllib.env import PettingZooEnv


class TokenizedMPE(PettingZooEnv):
    def __init__(
            self, 
            env, 
            render_mode,
            **kwargs,
        ):
        super().__init__(env)
        
        ## CONFIG
        self.env_name = str(env)
        self.id = hex(id(self))
        self.render_mode = render_mode
        self.metadata = {"render.modes": [render_mode], "render_fps": 20}

        ## BUILD OBS SPACE
        n = env.unwrapped.scenario.num_entities
        k = env.unwrapped.scenario.num_features
        env.unwrapped.observation_spaces = gym.spaces.Dict({
            agent_id: gym.spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(n, k),
                dtype=np.float32,
            )
            for agent_id in self._agent_ids
        })
        self.observation_space = gym.spaces.Dict(
            {agent_id: self.env.observation_space(agent_id) for agent_id in self._agent_ids}
        )
        return
    

    def step(self, action_dict:dict):
        obs_dict = {}
        rew_dict = {}
        term_dict = {}
        trun_dict = {}
        info_dict = {}
        for _ in self.env.agents:
            obs, reward, terminated, truncated, info = super().step({self.env.agent_selection: action_dict[self.env.agent_selection]})
            obs_dict.update(obs)
            rew_dict.update(reward)
            term_dict.update(terminated)
            trun_dict.update(truncated)
            info_dict.update(info)
        rew_dict = {k:v for k,v in rew_dict.items()}
        return obs_dict, rew_dict, term_dict, trun_dict, info_dict
    

    def reset(self, seed=None, options=None):
        info_dict = self.env.reset(seed=seed, options=options)
        obs_dict = {agent_id: self.env.observe(agent_id) for agent_id in self.env.agents}
        return obs_dict, info_dict or {}
    

    def render(self):
        return self.env.render()
