import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.stats import *

class _Segmentation(nn.Module):
    def __init__(self, backbone,classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
      
    def forward(self, x,mu_t_f1=0, std_t_f1=0, transfer=False,mix=False,activation=None):
        input_shape = x.shape[-2:]
        features = {}
        features['low_level'] = self.backbone(x,trunc1=False,trunc2=False,
           trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)

        if transfer:

            mean, std = calc_mean_std(features['low_level'])
            self.size = features['low_level'].size()

            features_low_norm = (features['low_level'] - mean.expand(
                self.size)) / std.expand(self.size)
            
            if mix:
                s = torch.rand((mean.shape[0],mean.shape[1])).to('cuda').unsqueeze(-1).unsqueeze(-1)
                mu_mix = s * mean + (1-s) * mu_t_f1
                std_mix = s * std + (1-s) * std_t_f1
                features['low_level'] = (std_mix.expand(self.size) * features_low_norm + mu_mix.expand(self.size))
            else:
                features['low_level'] = (std_t_f1.expand(self.size) * features_low_norm + mu_t_f1.expand(self.size))
            features['low_level'] = activation(features['low_level'])
                

        features['out'] = self.backbone(features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)

        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features