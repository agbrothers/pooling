import torch.nn as nn

    
class MLP(nn.Module):
    def __init__(self, 
            dim_input, 
            dim_hidden, 
            dim_output,
            num_layers=3,
            activation=nn.GELU(),
            dropout=0,
            **kwargs, 
        ):
        super().__init__()

        assert num_layers > 0

        if num_layers == 1:
            self.layers = nn.ModuleList(
                [nn.Linear(dim_input, dim_output)]
            )
        else:
            self.layers = nn.ModuleList(
                [nn.Linear(dim_input, dim_hidden)] + \
                [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers-2)] + \
                [nn.Linear(dim_hidden, dim_output)] 
            )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._dim_output = dim_output
        self._num_layers = num_layers
        self._recurrent = False
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x)
