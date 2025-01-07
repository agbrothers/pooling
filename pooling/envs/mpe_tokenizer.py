import numpy as np
import gymnasium as gym
from ray.rllib.env import PettingZooEnv


def reward_processor(pred_landmark_collision, pred_prey_collision):

    def adversary_reward(agent, world):
        # Predators are rewarded for collisions with prey
        rew = 0
        if not agent.collide:
            return rew

        prey_min = agent.size + world.agents[-1].size
        landmark_min = agent.size + world.landmarks[-1].size

        prey_pos = np.array([a.state.p_pos for a in world.agents if not a.adversary])
        landmarks_pos = np.array([lm.state.p_pos for lm in world.landmarks])

        ## COLLIDE WITH PREY 
        prey_dist = np.linalg.norm(prey_pos - agent.state.p_pos, axis=1)
        prey_collisions = prey_dist <= prey_min
        rew += sum(prey_collisions*pred_prey_collision)
        
        ## COLLIDE WITH LANDMARK
        landmarks_dist = np.linalg.norm(landmarks_pos - agent.state.p_pos, axis=1)
        landmark_collisions = landmarks_dist <= landmark_min
        rew += sum(landmark_collisions*pred_landmark_collision)
        return rew
    
    return adversary_reward


class TokenizedMPE(PettingZooEnv):
    def __init__(
            self, 
            env, 
            mpe_config, 
            landmark_spawning, 
            pred_landmark_collision=0,
            pred_prey_collision=10,
            log_path=None,
            log_agents=None,
            log_rew=False,
            log_obs=False,
            log_act=False,
            **kwargs,
        ):
        super().__init__(env)
        
        ## CONFIG
        env_name = str(env)
        self.id = hex(id(self))
        self.config = mpe_config
        self.landmark_spawning = landmark_spawning
        self.render_mode = mpe_config["render_mode"]
        self.metadata = {
            "render.modes": [self.render_mode],
            "render_fps": 20, 
        }

        ## LOGGING KWARGS
        self.log_path = log_path
        self.log_agents = log_agents or []
        self.log_rew = log_rew
        self.log_obs = log_obs
        self.log_act = log_act

        ## SELECT TOKENIZER BASED ON MPE SCENARIO
        if env_name == "simple_tag_v3":
            self.num_good = mpe_config["num_good"]
            self.num_obstacles = mpe_config["num_obstacles"]
            self.num_adversaries = mpe_config["num_adversaries"]
            self.dim_token = 5
            self.tokenizer = self.simple_tag_tokenizer
        elif env_name == "simple_cover_v3":
            pass

        ## OVERRIDE ADVERSARY REWARD FUNCTION
        if hasattr(env.unwrapped.scenario, "adversary_reward"):
            env.unwrapped.scenario.adversary_reward = reward_processor(
                pred_landmark_collision,
                pred_prey_collision,
            )

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
            self.episode_reward[aid] += reward[aid]
            
            ## OVERRIDE REWARD
            # self.env.unwrapped.world.agents

        obs_dict = self.tokenizer(obs_dict)
        rew_dict = {k:v for k,v in rew_dict.items()}
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
        if hasattr(self.env.world, "agents"):
            self.init_agents()
        if hasattr(self.env.world, "landmarks"):
            self.init_landmarks()
        obs_dict = {agent_id: self.env.observe(agent_id) for agent_id in self.env.agents}
        obs_dict = self.tokenizer(obs_dict)
        return obs_dict, info_dict or {}
    
    def init_agents(self):
        for agent in self.env.world.agents:
            spawn_dist = np.log2(len(self.env.world.landmarks)) 
            # agent.state.p_pos = np.random.normal(loc=spawn_dist, scale=spawn_dist, size=self.env.world.dim_p)            
            agent.state.p_pos = np.random.uniform(-spawn_dist, +spawn_dist, self.env.world.dim_p)
            agent.state.p_vel = np.random.uniform(-0.5, +0.5, self.env.world.dim_p)
            agent.state.c = np.zeros(self.env.world.dim_c)        
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            # agent.state.p_vel = np.zeros(self.env.world.dim_p)

    def init_landmarks(self):
        for i, landmark in enumerate(self.env.world.landmarks):
            if not landmark.boundary:
                ## ZERO VELOCITY
                landmark.state.p_vel = np.zeros(self.env.world.dim_p)        

                ## DEFAULT SPAWNING 
                if self.landmark_spawning == "default":
                    landmark.state.p_pos = np.random.uniform(-0.9, +0.9, self.env.world.dim_p)
                ## SPREAD
                elif self.landmark_spawning == "spread":
                    spawn_dist = np.log2(len(self.env.world.landmarks))
                    landmark.state.p_pos = np.random.uniform(-spawn_dist, +spawn_dist, self.env.world.dim_p)
                ## OUT-OF-BOUNDS
                elif self.landmark_spawning == "out_of_bounds":
                    noise = 2*np.random.rand() - 1
                    landmark.state.p_pos = np.ones(self.env.world.dim_p) * 6. + (noise*1e-3)
                # RADIAL
                elif self.landmark_spawning == "radial":
                    r = 8.
                    theta = 2*np.pi*np.random.rand()
                    landmark.state.p_pos = np.array((r*np.sin(theta), r*np.cos(theta)))
                ## ORIGIN
                elif self.landmark_spawning == "origin":
                    landmark.state.p_pos = np.zeros(self.env.world.dim_p)
                                

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
            obstacle_idx = 4

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
            # tokens[:, :4] *= 0.1

            token_dict[agent_id] = tokens

        return token_dict

    def get_episode_reward(self):
        return self.episode_reward

    def log(self, path:str, value:float):
        with open(path.replace(".csv", f"_{self.id}.csv"), "a") as file:
        # with open(path, "w") as file:
            file.write(f"{value}\n")                


