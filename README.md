# PODA (OBJECT DETECTION 🔥): Prompt-driven Zero-shot Domain Adaptation
[Mohammad Fahes<sup>1</sup>](https://mfahes.github.io/),
[Tuan-Hung Vu<sup>1,2</sup>](https://tuanhungvu.github.io/),
[Andrei Bursuc<sup>1,2</sup>](https://abursuc.github.io/),
[Patrick Pérez<sup>1,2</sup>](https://ptrckprz.github.io/),
[Raoul de Charette<sup>1</sup>](https://team.inria.fr/rits/membres/raoul-de-charette/)</br>
<sup>1</sup> Inria, Paris, France.

<sup>2</sup> valeo.ai, Paris, France.<br>

TL; DR: PØDA (or PODA) is a simple feature augmentation method for zero-shot domain adaptation guided by a single textual description of the target domain.

Project page: https://astra-vision.github.io/PODA/  
Paper: https://arxiv.org/abs/2212.03241

## Citation
```
@InProceedings{fahes2023poda,
  title={P{\O}DA: Prompt-driven Zero-shot Domain Adaptation},
  author={Fahes, Mohammad and Vu, Tuan-Hung and Bursuc, Andrei and P{\'e}rez, Patrick and de Charette, Raoul},
  booktitle={ICCV},
  year={2023}
}
```

# Getting started

## Clone repository

```bash
git clone -b detection git@github.com:astra-vision/PODA.git
cd PODA
```
    
## Preparing the environment
```bash
<PODA_root_dir>
conda create -n podadet pytorch==1.11.0 torchvision==0.12.0 pytorch-cuda=11.3 -c pytorch -c nvidia
conda activate podadet
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html --no-cache-dir
FORCE_CUDA=1 pip install --no-cache-dir -e .
```

## Preparing datasets
Download and organize [Cityscapes](https://www.cityscapes-dataset.com/), [Cityscapes Foggy](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) (attenuation coefficient of 0.02m-1) and [Diverse Weather Dataset](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B) in the following structure:

```bash
PODA_root_dir
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   └── val
│   │   └── annotations
│   │       ├── instancesonly_filtered_gtFine_train.json
│   │       └── instancesonly_filtered_gtFine_val.json
│   ├── cityscapes_foggy
│   │   └── val
│   │       ├── frankfurt
│   │       │   ├── frankfurt_000000_000294_leftImg8bit.png
│   │       │   └── ...
│   │       ├── lindau
│   │       └── munster
│   └── diverse_weather
│           ├── daytime_clear
│           ├── Night-Sunny
│           ├── night_rainy
│           ├── dusk_rainy
│           └── daytime_foggy
│   ...
```

## Downloading augmented features and pretrained checkpoints
[Download](https://github.com/astra-vision/PODA/releases/tag/v1.0.0) and place uncompressed files in the <PODA_root_dir> as follows:

```bash
PODA_root_dir
├── ...
├── augmented_feats
│   ├── OD_aug_iccv
│   └── fog_f1_templates_100it
├── checkpoints
│   ├── clip_visual_encoder_resnet50.pth
│   └── ...
├── ...
└── work_dirs
│   ├── faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes
│   │   └── latest.pth
│   │   
│   ├── faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes_PODA_fog
│   ├── faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3
│   ├── faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP
│   ├── faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_dayfog_lr4e-4_finetuneCLIP
│   ├── faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_duskrain_lr4e-4_finetuneCLIP
│   ├── faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_night_lr4e-4_finetuneCLIP
│   └── faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_nightrain_lr4e-4_finetuneCLIP
├── ...
```

## Training & Testing
### Src-domain Cityscapes + zero-shot adaptation to fog weather
```bash
# src-only
python tools/train.py ./configs/PODA/faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes.py --auto-scale-lr

# .. then train with PODA's augmented features of fog
python tools/train.py ./configs/PODA/faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes_PODA_fog.py --auto-scale-lr

# test on Cityscapes-Foggy
python tools/test.py ./configs/PODA/faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes_PODA_fog.py ./work_dirs/faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes_PODA_fog/latest.pth --eval bbox
```

### Src-domain Day-Clear + zero-shot adaptation to diverse weathers in DWD
```bash
# src-only (2 phases)
python tools/train.py ./configs/PODA/faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3.py --auto-scale-lr
python tools/train.py ./configs/PODA/faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP.py --auto-scale-lr

# .. then train with PODA's augmented features, e.g. night
python tools/train.py ./configs/PODA/faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_night_lr4e-4_finetuneCLIP.py --auto-scale-lr

# test on DWD, e.g. night
python tools/test.py ./configs/PODA/faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_night_lr4e-4_finetuneCLIP.py ./work_dirs/faster_rcnn_r101_fpn_1x_pretrainedCLIP_diverseweather_dayclearnew_lr4e-3_srconly_lr4e-4_finetuneCLIP_latest_PODA_night_lr4e-4_finetuneCLIP/latest.pth --eval mAP

```

### Results
We show some results of PØDA for object detection. The metric is mAP%.

❗❗❗ [08/07/2024] For the Night-Clear and Day-Foggy results, we corrected the evaluation bug from the original paper where the test split was mistakenly used instead of the train split for testing.

<p align="center">
  <img src="./images/PODA_for_OD.png/" style="width:75%"/>
</p>


# License
PØDA is released under the [Apache 2.0 license](./LICENSE).

# Acknowledgement
The codebase heavily depends on the [mmdetection v2.28.2](https://github.com/open-mmlab/mmdetection/releases/tag/v2.28.2) and uses code from [CLIP](https://github.com/openai/CLIP)
