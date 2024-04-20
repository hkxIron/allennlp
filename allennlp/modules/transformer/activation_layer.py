from typing import Union
import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule

from transformers.models.bert.modeling_bert import ACT2FN


"""
注意：这里的激活层都带了一个dense layer
"""
class ActivationLayer(TransformerModule, FromParams):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Union[str, torch.nn.Module],
        pool: bool = False,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(in_features=hidden_size, out_features=intermediate_size)
        if isinstance(activation, str):
            self.act_fn = ACT2FN[activation]
        else:
            self.act_fn = activation
        self.pool = pool

    def get_output_dim(self) -> int:
        return self.dense.out_features

    def forward(self, hidden_states):
        if self.pool:
            hidden_states = hidden_states[:, 0]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states
