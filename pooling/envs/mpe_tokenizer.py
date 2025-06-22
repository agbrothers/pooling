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
        self.scripted_agents = [a.name for a in self.env.world.scripted_agents]
        self.scripted_agent_dummy_actions = {a:self.action_space[a].sample() for a in self.scripted_agents}
        return
    

    def step(self, action_dict:dict):
        obs_dict = {}
        rew_dict = {}
        info_dict = {}
        term_dict = {"__all__": False}
        trun_dict = {"__all__": False}
        action_dict.update(self.scripted_agent_dummy_actions)
        for _ in self.env.agents:
            obs, reward, terminated, truncated, info = super().step(action_dict)
            agent = self.env.agent_selection
            if agent in self.scripted_agents:
                continue
            obs_dict[agent] = obs[agent]
            rew_dict[agent] = reward[agent]
            info_dict[agent] = info[agent]
            term_dict[agent] = terminated[agent]
            trun_dict[agent] = truncated[agent]
            term_dict["__all__"] ^= terminated["__all__"]
            trun_dict["__all__"] ^= truncated["__all__"]
        return obs_dict, rew_dict, term_dict, trun_dict, info_dict
    

    def reset(self, seed=None, options=None):
        info_dict = self.env.reset(seed=seed, options=options)
        obs_dict = {a.name: self.env.observe(a.name) for a in self.env.world.agents if a.name not in self.scripted_agents}
        return obs_dict, info_dict or {}
    

    def render(self):
        return self.env.render()
