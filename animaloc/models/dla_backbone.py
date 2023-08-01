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
    ''' HerdNet architecture '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64
        ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            num_classes (int, optional): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16. 
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end 
                of decoder. Defaults to 64.
        '''

        super(DLAEncoder, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        base_name = 'dla{}'.format(num_layers)

        self.down_ratio = down_ratio
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

        self.cls_head = nn.Linear(512*16*16, 1) # binary head
        
    def forward(self, input: torch.Tensor):

        encode = self.base_0(input) # 1x512x16x16
        bottleneck = self.bottleneck_conv(encode[-1])
        bottleneck= torch.reshape(bottleneck, (bottleneck.size()[0],-1)) # not sure if it is the right approach
        encode[-1] = bottleneck # 1x512x16x16
        cls = self.cls_head(encode[-1])
        
        #cls = nn.functional.softmax(cls)
        return cls
    
    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    
    def reshape_classes(self, num_classes: int) -> None:
        ''' Reshape architecture according to a new number of classes.

        Arg:
            num_classes (int): new number of classes
        '''
        
        self.cls_head[-1] = nn.Conv2d(
                self.head_conv, num_classes, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                )

        self.cls_head[-1].bias.data.fill_(0.00)

        self.num_classes = num_classes