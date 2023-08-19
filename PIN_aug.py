################
#script to augment features with CLIP

import pickle
import os
import clip
import torch
import network
import torch.nn as nn
from utils.stats import calc_mean_std
import argparse
from main import get_dataset
from torch.utils import data
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to dataset")
    parser.add_argument("--save_dir", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','gta5'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default = 'RN50',
                        help= "backbone name" )
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type = int, default =100,
                        help= "total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False,
                        help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # target domain description
    parser.add_argument("--domain_desc", type=str , default = "driving at night.",
                        help = "description of the target domain")


    return parser



class PIN(nn.Module):
    def __init__(self,shape,content_feat):
        super(PIN,self).__init__()
        self.shape = shape
        self.content_feat = content_feat.clone().detach()
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)
        self.size = self.content_feat.size()
        self.content_feat_norm = (self.content_feat - self.content_mean.expand(
        self.size)) / self.content_std.expand(self.size)

        self.style_mean =   self.content_mean.clone().detach() 
        self.style_std =   self.content_std.clone().detach()

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True)
        self.style_std = nn.Parameter(self.style_std, requires_grad = True)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        
        self.style_std.data.clamp_(min=0)
        target_feat =  self.content_feat_norm * self.style_std.expand(self.size) + self.style_mean.expand(self.size)
        target_feat = self.relu(target_feat)
        return target_feat

def main():

    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,data_aug=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19,BB= opts.BB,replace_stride_with_dilation=[False,False,False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56,56))
    else:
        t1 = lambda x:x

    #text
    #target text
    target = compose_text_with_templates(opts.domain_desc, imagenet_templates)

    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size,1).type(torch.float32)  # (B,1024)

    for i,(img_id, tar_id, images, labels) in enumerate(train_loader):
            print(i)    
            
            f1 = model.backbone(images.to(device),trunc1=False,trunc2=False,
            trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)  # (B,C1,H1,W1)
            
            #optimize mu and sigma of target features with CLIP
            model_pin_1 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
            model_pin_1.to(device)


            optimizer_pin_1 = torch.optim.SGD(params=[
                {'params': model_pin_1.parameters(), 'lr': 1},
            ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)

            if i == len(train_loader)-1 and f1.shape[0] < opts.batch_size :
                text_target = text_target[:f1.shape[0]]

            while cur_itrs< opts.total_it: 

                cur_itrs += 1
                if cur_itrs % opts.total_it==0:
                    print(cur_itrs)

                optimizer_pin_1.zero_grad()
            
                f1_hal = model_pin_1()
                f1_hal_trans = t1(f1_hal)

                #target_features (optimized)
                target_features_from_f1 = model.backbone(f1_hal_trans,trunc1=True,trunc2=False,trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=False)
                target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
       
                #loss
                loss_CLIP1 = (1- torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()

                writer.add_scalar("loss_CLIP_f1"+str(i),loss_CLIP1,cur_itrs)
               
                loss_CLIP1.backward(retain_graph=True)
              
                optimizer_pin_1.step()
        
            cur_itrs = 0
            
            for name, param in model_pin_1.named_parameters():
                if param.requires_grad and name == 'style_mean':
                    learnt_mu_f1 = param.data
                elif param.requires_grad and name == 'style_std':
                    learnt_std_f1 = param.data

            for k in range(learnt_mu_f1.shape[0]):
                learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
                learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

                stats = {}
                stats['mu_f1'] = learnt_mu_f1_
                stats['std_f1'] = learnt_std_f1_

                with open(opts.save_dir+'/'+img_id[k].split('/')[-1]+'.pkl', 'wb') as f:
                    pickle.dump(stats, f)
     
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)

main()