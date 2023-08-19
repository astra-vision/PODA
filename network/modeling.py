from ._deeplab import  DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet_clip
import os
import torch

def deeplabv3plus_resnet_clip(num_classes=19,BB = "RN50",replace_stride_with_dilation=[False,False,True]):
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
   
    model_url = "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
    
    model_path = resnet_clip._download(model_url, os.path.expanduser("~/.cache/clip"))
    with open(model_path, 'rb') as opened_file:
        backbone = torch.jit.load(opened_file, map_location="cpu").eval()
        backbone = resnet_clip.build_model(backbone.state_dict(),replace_stride_with_dilation=replace_stride_with_dilation).to(device)

    inplanes = 2048
    low_level_planes = 256
    aspp_dilate = [6,12,18]

    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = backbone.visual
    model = DeepLabV3(backbone,classifier)
    return model