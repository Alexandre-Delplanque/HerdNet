__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 29, 2023
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.2.0"


import torch

import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from typing import Optional

from .register import MODELS

from . import dla as dla_modules


@MODELS.register()
class DLAEncoder(nn.Module):
    ''' DLA encoder architecture '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            num_classes (int, optional): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
        '''

        super(DLAEncoder, self).__init__()
        
        base_name = 'dla{}'.format(num_layers)

        self.num_classes = num_classes
        self.head_conv = head_conv

        self.first_level = int(np.log2(down_ratio))

        # backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)

        channels = self.channels_0


        # bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1], 
            kernel_size=1, stride=1, 
            padding=0, bias=True
        )
        self.pooling= nn.AvgPool2d(kernel_size= 16, stride=1, padding=0) # we take the average of each filter
        self.cls_head = nn.Linear(512, 1) # binary head
        
    def forward(self, input: torch.Tensor):

        encode = self.base_0(input) # Nx512x16x16
        bottleneck = self.bottleneck_conv(encode[-1])
        bottleneck = self.pooling(bottleneck)
        bottleneck= torch.reshape(bottleneck, (bottleneck.size()[0],-1)) # keeping the first dimension (samples)
        encode[-1] = bottleneck # Nx512
        cls = self.cls_head(encode[-1])
        
        #cls = nn.functional.sigmoid(cls)
        return cls
    
    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    
