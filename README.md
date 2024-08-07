🔥[08/07/2024]🚀 the detection of PODA was release in the [detection branch](https://github.com/astra-vision/PODA/tree/detection)

# PODA: Prompt-driven Zero-shot Domain Adaptation
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
# Overview
<p align="center">
  <b>Overview of PØDA</b>
</p>
<p align="center">
  <img src="./images/teaser.png/" style="width:100%"/>
</p>

# Method
<p align="center">
  <b>We propose Prompt-driven Instance Normalization (PIN) to augment feature styles based on "feature/target domain description" similarity</b>
</p>
<p align="center">
  <img src="./images/method.png/" style="width:80%"/>
</p>

# Teaser
<p align="center">
  <b>Test on unseen youtube video of night driving:<br />
  Training dataset: Cityscapes <br />
  Prompt: "driving at night"
  </b>
</p>
<p align="center">
  <img src="./images/night_video.gif" style="width:100%"/>
</p>

# Table of Content
- [News](#news)
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Source models](#source-models)
- [Running PODA](#running-poda)
  - [Source training](#source-training)
  - [Feature optimization](#feature-optimization)
  - [Model adaptation](#model-adaptation)
  - [Evaluation](#evaluation)
- [Inference & Visualization](#inference--visualization)
- [Qualitative Results](#qualitative-results)
- [PODA for Object Detection](#poda-for-object-detection)
- [License](#license)
- [Acknowledgement](#acknowledgement)

# News
* 29/11/2023: Check out our recent work [A Simple Recipe for Language-guided Domain Generalized Segmentation](https://arxiv.org/pdf/2311.17922.pdf).
* 19/08/2023: Camera-ready version is on [arxiv](https://arxiv.org/pdf/2212.03241.pdf).
* 14/07/2023: PODA is accepted at ICCV 2023.

# Installation
## Dependencies

First create a new conda environment with the required packages:
```
conda env create --file environment.yml
```

Then activate environment using:
```
conda activate poda_env
```

## Datasets

* **CITYSCAPES**: Follow the instructions in [Cityscapes](https://www.cityscapes-dataset.com/)
  to download the images and semantic segmentation ground-truths. Please follow the dataset directory structure:
  ```html
  <CITYSCAPES_DIR>/             % Cityscapes dataset root
  ├── leftImg8bit/              % input image (leftImg8bit_trainvaltest.zip)
  └── gtFine/                   % semantic segmentation labels (gtFine_trainvaltest.zip)
  ```

* **ACDC**: Download ACDC images and ground truths from [ACDC](https://acdc.vision.ee.ethz.ch/download). Please follow the dataset directory structure:
  ```html
  <ACDC_DIR>/                   % ACDC dataset root
  ├── rbg_anon/                 % input image (rgb_anon_trainvaltest.zip)
  └── gt/                       % semantic segmentation labels (gt_trainval.zip)
  ```
 
* **GTA5**: Download GTA5 images and ground truths from [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/). Please follow the dataset directory structure:
  ```html
  <GTA5_DIR>/                   % GTA5 dataset root
  ├── images/                   % input image 
  └── labels/                   % semantic segmentation labels
  ```
## Source models
The source models are available [here](https://drive.google.com/drive/folders/15-NhVItiVbplg_If3HJibokJssu1NoxL?usp=sharing).

# Running PODA

## Source training
```
python3 main.py \
  --dataset <source_dataset> \
  --data_root <path_to_source_dataset> \
  --data_aug \
  --lr 0.1 \
  --crop_size 768 \
  --batch_size 2 \
  --freeze_BB \
  --ckpts_path saved_ckpts
```

## Feature optimization
```
python3 PIN_aug.py \
--dataset <source_dataset> \
--data_root <path_to_source_dataset> \
--total_it 100 \
--resize_feat \
--domain_desc <target_domain_description>  \
--save_dir <directory_for_saved_statistics>
```

## Model adaptation
``` 
python3 main.py \
--dataset <source_dataset> \
--data_root <path_to_source_dataset> \
--ckpt <path_to_source_checkpoint> \
--batch_size 8 \
--lr 0.01 \
--ckpts_path adapted \
--freeze_BB \
--train_aug \
--total_itrs 2000 \ 
--path_mu_sig <path_to_augmented_statistics>
```

## Evaluation
```
python3 main.py \
--dataset <dataset_name> \
--data_root <dataset_path> \
--ckpt <path_to_tested_model> \
--test_only \
--val_batch_size 1 \
--ACDC_sub <ACDC_subset_if_tested_on_ACDC>   
```

# Inference & Visualization
To test any model on any image and visualize the output, please add the images to predict_test directory and run:
``` 
python3 predict.py \
--ckpt <ckpt_path> \
--save_val_results_to <directory_for_saved_output_images>
```

# Qualitative Results
<p align="center">
  <b>PØDA for uncommon driving situations</b>
</p>
<p align="center">
  <img src="./images/uncommon.png/" style="width:75%"/>
</p>

# PODA for Object Detection
Our feature augmentation is task-agnostic, as it operates on the feature extractor's level. We show some results of PØDA for object detection. The metric is mAP%

![08/07/2024] For the Night-Clear and Day-Foggy results, we corrected the evaluation bug from the original paper where the test split was mistakenly used instead of the train split for testing.

<p align="center">
  <img src="./images/PODA_for_OD.png/" style="width:75%"/>
</p>

# License
PØDA is released under the [Apache 2.0 license](./LICENSE).

# Acknowledgement
The code heavily borrows from this implementation of [DeepLabv3+](https://github.com/VainF/DeepLabV3Plus-Pytorch), and uses code from [CLIP](https://github.com/openai/CLIP)

---

[↑ back to top](#poda-prompt-driven-zero-shot-domain-adaptation)
