import numpy as np
import gymnasium as gym
from ray.rllib.env import PettingZooEnv


def reward_processor(pred_landmark_collision, pred_prey_collision):

    def adversary_reward(agent, world):
        # Predators are rewarded for collisions with prey
        rew = 0
        if not agent.collide:
            return rew

        ## COLLIDE WITH PREY 
        if pred_prey_collision != 0:
            prey_min = agent.size + world.agents[-1].size
            prey_pos = np.array([a.state.p_pos for a in world.agents if not a.adversary])
            prey_dist = np.linalg.norm(prey_pos - agent.state.p_pos, axis=1)
            prey_collisions = prey_dist <= prey_min
            rew += sum(prey_collisions*pred_prey_collision)
        
        ## COLLIDE WITH LANDMARK
        if pred_landmark_collision != 0 and len(world.landmarks) > 0:
            landmark_min = agent.size + world.landmarks[-1].size
            landmarks_pos = np.array([lm.state.p_pos for lm in world.landmarks])
            landmarks_dist = np.linalg.norm(landmarks_pos - agent.state.p_pos, axis=1)
            landmark_collisions = landmarks_dist <= landmark_min
            rew += sum(landmark_collisions*pred_landmark_collision)
        return rew
    
    return adversary_reward


def observation(agent, world):
    own_role  = 0
    lm_role   = 1
    pred_role = 2
    prey_role = 3
    own_token = np.hstack((agent.state.p_pos, agent.state.p_vel, [own_role]))
    tokens = [own_token]

    for lm in world.landmarks:
        if not lm.boundary:
            tokens.append(
                np.hstack((lm.state.p_pos, lm.state.p_vel, [lm_role]))
            )        
    
    for entity in world.agents:
        if entity is agent: 
            continue
        role = pred_role if entity.adversary else prey_role
        tokens.append(
            np.hstack((entity.state.p_pos, entity.state.p_vel, [role]))
        )  
    tokens = np.stack(tokens)
    ## SHIFT INTO RELATIVE COORDINATE FRAME
    tokens[:, :4] -= tokens[0, :4]
    # tokens[:, :4] /= 5
    return tokens


class TokenizedMPE(PettingZooEnv):
    def __init__(
            self, 
            env, 
            mpe_config, 
            landmark_spawning, 
            pred_landmark_collision=0,
            pred_prey_collision=10,
            collidable_landmarks=True,
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
            self.num_tokens = self.num_good + self.num_obstacles + self.num_adversaries
            # self.tokenizer = self.simple_tag_tokenizer
        elif env_name == "simple_cover_v3":
            pass

        ## OVERRIDE OBSERVATION FUNCTION
        if hasattr(env.unwrapped.scenario, "observation"):
            env.unwrapped.scenario.observation = observation
        ## OVERRIDE ADVERSARY REWARD FUNCTION
        if hasattr(env.unwrapped.scenario, "adversary_reward"):
            env.unwrapped.scenario.adversary_reward = reward_processor(
                pred_landmark_collision,
                pred_prey_collision,
            )

        ## BUILD OBS SPACE
        env.unwrapped.observation_spaces = gym.spaces.Dict({
            agent_id: gym.spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(self.num_tokens, self.dim_token),
                dtype=np.float32,
            )
            for agent_id in self._agent_ids
        })
        self.observation_space = gym.spaces.Dict(
            {agent_id: self.env.observation_space(agent_id) for agent_id in self._agent_ids}
        )

        ## OVERRIDE LANDMARK COLLISION STATUS
        if not collidable_landmarks:
            for landmark in env.unwrapped.world.landmarks:
                landmark.collide = False        
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
        if hasattr(self.env.world, "agents"):
            self.init_agents()
        if hasattr(self.env.world, "landmarks"):
            self.init_landmarks()
        obs_dict = {agent_id: self.env.observe(agent_id) for agent_id in self.env.agents}
        return obs_dict, info_dict or {}
    
    
    def init_agents(self):
        spawn_min = 0.1
        spawn_max = np.log(len(self.env.world.agents)) /  np.log(4)
        spawn_dist = np.random.uniform(low=spawn_min, high=spawn_max)
        for agent in self.env.world.agents:   
            agent.state.p_pos = spawn_dist * np.random.uniform(-1., 1., self.env.world.dim_p)
            agent.state.p_vel = np.random.uniform(-0.5, +0.5, self.env.world.dim_p)
            agent.state.c = np.zeros(self.env.world.dim_c)        
        return


    def init_landmarks(self):
        if len(self.env.world.landmarks) == 0:
            return
        spawn_dist = max(np.log(len(self.env.world.landmarks)) /  np.log(4), 2.0)
        for i, landmark in enumerate(self.env.world.landmarks):
            if not landmark.boundary:
                ## ZERO VELOCITY
                landmark.state.p_vel = np.zeros(self.env.world.dim_p)        

                ## DEFAULT SPAWNING 
                if self.landmark_spawning == "default":
                    landmark.state.p_pos = np.random.uniform(-0.9, +0.9, self.env.world.dim_p)
                ## SPREAD
                elif self.landmark_spawning == "spread":
                    landmark.state.p_pos = spawn_dist * np.random.uniform(-1., 1., self.env.world.dim_p)
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
        return
                                

    def render(self):
        return self.env.render()
