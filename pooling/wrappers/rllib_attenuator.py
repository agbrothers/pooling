from torch import Tensor
from typing import List, Tuple
from pooling.wrappers.rllib_wrapper import RLlibWrapper
from pooling.wrappers.actor_critic import TokenActorCritic
from pooling.models.attenuator import Attenuator


class RLlibAttenuator(RLlibWrapper):
    '''
    RLlib wrapper for the Attenuator 
    '''    
    def __init__(self, *args, **kwargs):
        if "model" not in kwargs:
            kwargs.update({"model": TokenActorCritic})
        if "pi" not in kwargs:
            kwargs.update({"pi": Attenuator})
        if "vf" not in kwargs:
            kwargs.update({"vf": Attenuator})
        super().__init__(*args, **kwargs)
        return

    # @override(ModelV2)
    # def forward(
    #     self,
    #     input_dict: Dict[str, TensorType],
    #     state: List[TensorType],
    #     seq_lens: TensorType,
    # ) -> Tuple[TensorType, List[TensorType]]:
    #     """Adds time dimension to batch before sending inputs to forward_rnn().

    #     You should implement forward_rnn() in your subclass."""
    #     flat_inputs = input_dict["obs_flat"].float()
    #     # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
    #     # as input_dict may have extra zero-padding beyond seq_lens.max().
    #     # Use add_time_dimension to handle this
    #     self.time_major = self.model_config.get("_time_major", False)
    #     inputs = add_time_dimension(
    #         flat_inputs,
    #         seq_lens=seq_lens,
    #         framework="torch",
    #         time_major=self.time_major,
    #     )
    #     output, new_state = self.forward_rnn(inputs, state, seq_lens)
    #     output = torch.reshape(output, [-1, self.num_outputs])
    #     return output, new_state

    def forward_rnn(
        self, inputs: Tensor, state: List[Tensor], seq_lens: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        
        return self.forward(inputs)

    def set_recurrent_state(self, input_dict):
        ##  OVERWRITE THE RECURRENT STATE FROM RLLIB BUFFERS
        if not self.recurrent: return 
        self.model.pi.set_recurrent_state(input_dict["state_in_0"])
        if self.model.vf:
            self.model.vf.set_recurrent_state(input_dict["state_in_1"])
        return


    def get_output_state(self):        
        ##  RETURN THE MOST RECENT STATE FOR RLLIB TO TRACK IN BUFFERS
        if not self.recurrent: return []
        pi_output_state = [self.model.pi.get_recurrent_state()] 
        vf_output_state = [self.model.vf.get_recurrent_state()] if self.model.vf else []
        return pi_output_state + vf_output_state 
