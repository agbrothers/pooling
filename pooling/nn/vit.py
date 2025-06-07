import math
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from pooling.models.attenuator import Attenuator

""" NOTE:
    DERIVED FROM Phil Wang's ViT IMPLEMENTATION
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py 
"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


class ViT(nn.Module):
    
    def __init__(self, size_img, size_patch, dim_hidden, num_classes, num_layers,  dropout_embd=0.,  channels=3, seed=None, **kwargs):
        super().__init__()
        image_height, image_width = pair(size_img)
        patch_height, patch_width = pair(size_patch)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        dim_patch = channels * patch_height * patch_width

        self.size_img = size_img
        self.size_patch = size_patch
        self.dim_patch = size_patch
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(dim_patch),
            nn.Linear(dim_patch, dim_hidden),
            nn.LayerNorm(dim_hidden)
        )

        self.pos_embd  = nn.Parameter(torch.randn(1, num_patches, dim_hidden))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        self.dropout = nn.Dropout(dropout_embd)
        self.attenuator = Attenuator(dim_hidden, num_layers=num_layers, **kwargs)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, num_classes),
        ) 
        
        ## INITIALIZE WEIGHTS
        if seed:
            self.apply(self.initialize)
            for name,param in self.named_parameters():
                if name.endswith("out.weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * num_layers))        
        return
    
    def img_to_tokens(self, img):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embd
        return self.dropout(x)

    def forward(self, img):
        x = self.img_to_tokens(img)        
        x = self.attenuator(x)
        return self.classifier(x)
    
    def initialize(self, module) -> None:
        """ 
        INITIALIZATION SCHEME AS IN 
        [1] https://arxiv.org/pdf/1502.01852.pdf
        [2] https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L163
        
        """
        ## LINEAR LAYERS
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        ## LAYERNORMS
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        ## EMBEDDING WEIGHTS
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        return


if __name__ == "__main__":

    model = ViT(
        size_img = 256,
        size_patch = 32,
        dim_hidden = 1024,
        dim_ff = 2048,
        num_classes = 1000,
        num_layers = 6,
        num_heads = 16,
        dropout = 0.1,
        dropout_embd = 0.1,
    )

    img = torch.randn(2, 3, 256, 256)

    preds = model(img)
    assert preds.shape == (2, 1000), 'correct logits outputted'    
