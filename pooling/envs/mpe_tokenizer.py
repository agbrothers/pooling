import numpy as np

import gymnasium as gym

from ray.rllib.env import PettingZooEnv


def symlog(x):
    return np.sign(x) * np.log(np.abs(x)+1)

class TokenizedMPE(PettingZooEnv):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        
        ## CONFIG
        env_name = str(env)
        self.id = hex(id(self))
        self.config = env.unwrapped._ezpickle_kwargs
        self.render_mode = self.config["render_mode"]
        self.metadata = {
            "render.modes": [self.render_mode],
            "render_fps": 20, 
        }

        ## LOGGING KWARGS
        self.log_path = kwargs.get("log_path", None)
        self.log_agents = kwargs.get("log_agents", None)
        self.log_rew = kwargs.get("log_rew", False)
        self.log_obs = kwargs.get("log_obs", False)
        self.log_act = kwargs.get("log_act", False)

        ## SELECT TOKENIZER BASED ON MPE SCENARIO
        if env_name == "simple_tag_v3":
            self.num_good = self.config["num_good"]
            self.num_obstacles = self.config["num_obstacles"]
            self.num_adversaries = self.config["num_adversaries"]
            self.dim_token = 5
            self.tokenizer = self.simple_tag_tokenizer
        elif env_name == "simple_cover_v3":
            pass

        ## COMPUTE NEW OBS SPACE
        sample_obs = {aid:np.zeros((s.shape)) for aid,s in env.unwrapped.observation_spaces.items()}
        sample_tokens = self.tokenizer(sample_obs)

        env.unwrapped.observation_spaces = gym.spaces.Dict({
            aid: gym.spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=sample_tokens[aid].shape,
                dtype=np.float32,
            )
            for aid in self._agent_ids
        })
        self.observation_space = gym.spaces.Dict(
            {aid: self.env.observation_space(aid) for aid in self._agent_ids}
        )
        self.episode_reward = {aid:0. for aid in sorted(self._agent_ids)}
        # self.prev_episode_reward = None
        return
    

    def step(self, action_dict:dict):
        obs_dict = {}
        rew_dict = {}
        term_dict = {}
        trun_dict = {}
        info_dict = {}
        ## TODO: MAKE SURE __all__ is preserved per step
        for _ in self.env.agents:
            obs, reward, terminated, truncated, info = super().step({self.env.agent_selection: action_dict[self.env.agent_selection]})
            obs_dict.update(obs)
            rew_dict.update(reward)
            term_dict.update(terminated)
            trun_dict.update(truncated)
            info_dict.update(info)
            aid = list(reward.keys())[0]
            self.episode_reward[aid] += symlog(reward[aid])
        obs_dict = self.tokenizer(obs_dict)
        rew_dict = {k:symlog(v) for k,v in rew_dict.items()}
        return obs_dict, rew_dict, term_dict, trun_dict, info_dict
    
    def reset(self, seed=None, options=None):
        full_episode = self.env.unwrapped.steps >= self.env.unwrapped.max_cycles
        # if full_episode:
        #     print(self.episode_reward)
        if self.log_rew and full_episode:
            for agent_id in self.log_agents:
                self.log(self.log_path, self.episode_reward[agent_id])
        if self.log_obs and full_episode:
            pass
        if self.log_act and full_episode:
            pass
            
        # self.prev_episode_reward = self.episode_reward.copy()
        self.episode_reward = {aid:0. for aid in sorted(self._agent_ids)}
        info_dict = self.env.reset(seed=seed, options=options)
        obs_dict = {agent_id: self.env.observe(agent_id) for agent_id in self.env.agents}
        obs_dict = self.tokenizer(obs_dict)
        return obs_dict, info_dict or {}

    def render(self):
        return self.env.render()

    def simple_tag_tokenizer(self, obs_dict):

        """
        ORIGINAL OBS COMPUTED IN
        
        pettingzoo/mpe/simle_tag/simple_tag.py -> Scenario.observation()

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )

        """

        token_dict = {}
        for agent_id, obs in obs_dict.items():
            
            ## DETERMINE OBSERVER TYPE
            is_good = "agent" in agent_id
            num_good = self.num_good - is_good
            num_adversaries = self.num_adversaries - (not is_good)

            ## PROCESS ADVERSARY OBS
            # own_pos = obs[2:4]
            own_pos = [0,0]
            own_vel = obs[0:2]
            own_role = 0
            tokens = np.array([[ *own_pos, *own_vel, own_role ]])

            if self.num_obstacles > 0:
                obstacle_idx = 4 + 2*self.num_obstacles
                obstacle_pos = obs[4:obstacle_idx].reshape(self.num_obstacles, -1)
                obstacle_vel = np.zeros((self.num_obstacles, 2))
                obstacle_role = np.ones((self.num_obstacles, 1))
                obstacle_tokens = np.hstack((obstacle_pos, obstacle_vel, obstacle_role))
                tokens = np.vstack((tokens, obstacle_tokens))
            
            if num_adversaries > 0:
                adversary_idx = obstacle_idx + 2*num_adversaries
                adversary_pos = obs[obstacle_idx:adversary_idx].reshape(num_adversaries, -1)
                adversary_vel = np.zeros((num_adversaries, 2)) # np.tile(own_vel, (num_adversaries, 1))
                adversary_role = 2 * np.ones((num_adversaries, 1)) 
                adversary_tokens = np.hstack((adversary_pos, adversary_vel, adversary_role))
                tokens = np.vstack((tokens, adversary_tokens))

            if num_good > 0:
                good_pos_idx = adversary_idx + 2*num_good
                good_pos = obs[adversary_idx:good_pos_idx].reshape(num_good, -1)
                good_vel_idx = good_pos_idx + 2*num_good
                good_vel = obs[good_pos_idx:good_vel_idx].reshape(num_good, -1)
                good_role = 3 * np.ones((num_good, 1))
                good_tokens = np.hstack((good_pos, good_vel, good_role))
                tokens = np.vstack((tokens, good_tokens))

            tokens[:, 2:4] -= own_vel

            token_dict[agent_id] = tokens

        return token_dict

    def get_episode_reward(self):
        return self.episode_reward

    def log(self, path:str, value:float):
        with open(path.replace(".csv", f"_{self.id}.csv"), "a") as file:
        # with open(path, "w") as file:
            file.write(f"{value}\n")                