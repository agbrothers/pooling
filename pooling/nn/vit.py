""" DERIVED FROM Phil Wang's ViT IMPLEMENTATION
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py 
"""

import math
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from pooling.nn.transformer import Transformer
from pooling.nn.initialize import transformer_init


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
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(dim_patch),
            nn.Linear(dim_patch, dim_hidden),
            nn.LayerNorm(dim_hidden)
        )

        self.pos_embd  = nn.Parameter(torch.randn(1, num_patches + 1, dim_hidden))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        self.dropout = nn.Dropout(dropout_embd)
        self.transformer = Transformer(dim_hidden, num_layers=num_layers, **kwargs)
        self.classifier = nn.Linear(dim_hidden, num_classes)
        
        ## INITIALIZE WEIGHTS
        if seed:
            self.apply(transformer_init)
            for name,param in self.named_parameters():
                if name.endswith("out.weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * num_layers))        
        return
    
    def img_to_tokens(self, img):
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
        x = torch.cat((cls_tokens, x), dim = 1)

        x += self.pos_embd
        x = self.dropout(x)
        return x

    def forward(self, img):
        x = self.img_to_tokens(img)        

        x = self.transformer(x)

        cls_tokens = x[:, 0]
        return self.classifier(cls_tokens)
    

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
        dropout_emb = 0.1,
    )

    img = torch.randn(1, 3, 256, 256)

    preds = model(img)
    assert preds.shape == (1, 1000), 'correct logits outputted'    
