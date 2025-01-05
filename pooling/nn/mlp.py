import torch
import torch.nn as nn

    
class MLP(nn.Module):
    def __init__(self, 
            dim_input, 
            dim_hidden, 
            dim_output,
            num_layers=3,
            activation=nn.GELU(),
            dropout=0,
            seed=None,
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

        if seed:
            torch.manual_seed(seed)
            self.apply(self.initialize)

    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x)


    def initialize(self, module) -> None:
        """ 
        INITIALIZATION SCHEME AS IN 
        [1] https://arxiv.org/pdf/1502.01852.pdf
        [2] https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L163
        
        """
        ## LINEAR LAYERS
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        return
