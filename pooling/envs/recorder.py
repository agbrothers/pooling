"""Wrapper for recording videos."""
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.logger import set_level
from ray.rllib.env import MultiAgentEnv
from typing import Callable
set_level(40)

class RecordVideoMultiAgent(RecordVideo, MultiAgentEnv):
    """
    AUGMENT RecordVideo BASE CLASS TO HANDLE DICTIONARY OUTPUTS FROM MULTI-AGENT ENVIRONMENTS

    Refer to base class for video recording details. 

    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int],
        bool] = None,
        step_trigger: Callable[[int],
        bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False
    ):
        super().__init__(env, video_folder, episode_trigger, step_trigger, video_length, name_prefix, disable_logger)
        self.name_prefix = name_prefix + f"_{hex(id(self))}"
        ## TODO: Add reward to name prefix -> rename on close perhaps

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            obs_dict,
            rew_dict,
            term_dict,
            trun_dict,
            info_dict,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            if not self.is_vector_env:
                if term_dict["__all__"] or trun_dict["__all__"]:
                    self.episode_id += 1
                    self.terminated = term_dict["__all__"]
                    self.truncated = trun_dict["__all__"]
            elif term_dict["__all__"][0] or trun_dict["__all__"][0]:
                self.episode_id += 1
                self.terminated = term_dict["__all__"][0]
                self.truncated = trun_dict["__all__"][0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if term_dict["__all__"] or trun_dict["__all__"]:
                            self.close_video_recorder()
                    elif term_dict["__all__"][0] or trun_dict["__all__"][0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return obs_dict, rew_dict, term_dict, trun_dict, info_dict

    def close(self):
        super().close()
        return 
    