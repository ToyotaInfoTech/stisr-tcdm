# Scene Text Image Super-resolution based on Text-conditional Diffusion Models
[![arXiv](https://img.shields.io/badge/arXiv-2311.09759-b31b1b.svg)](https://arxiv.org/abs/2311.09759)

This is the official repogitory of the WACV2024 paper ["Scene Text Image Super-resolution based on Text-conditional Diffusion Models"](https://arxiv.org/abs/2311.09759).
This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

# Pre-trained models for DiMSS and GT-DiMSS

We are going to release checkpoints of the models and two generated dataset, SynTZ and SynSTR, for the main results in the paper.

# Model Tranining

## Requirements

To get started, install the required python packages using the following command:
```
pip install -e .
```

## Dataset
Downoad the TextZoom dataset from 
```
https://github.com/JasonBoy1/TextZoom
```

The structure of ``dataset`` directory is 
```
dataset
`-- TextZoom
    |-- test
    |   |-- easy
    |   |   |-- data.mdb
    |   |   `-- lock.mdb
    |   |-- hard
    |   |   |-- data.mdb
    |   |   `-- lock.mdb
    |   `-- medium
    |       |-- data.mdb
    |       `-- lock.mdb
    |-- train1
    |   |-- data.mdb
    |   `-- lock.mdb
    `-- train2
        |-- data.mdb
        `-- lock.mdb
```

## Pretrained recognizers
Download pretrained recognizers (CRNN, ASTER, MORAN).

CRNN:
```
https://github.com/meijieru/crnn.pytorch
```

ASTER:
```
https://github.com/ayumiymk/aster.pytorch  
```

MORAN:
```
https://github.com/Canjie-Luo/MORAN_v2
```

## Training
To train DiMSS on the TextZoom dataset, run the script via
```
bash train_dimss_textzoom.sh
```
Also, use the following script to train GT-DiMSS
```
bash train_gt_dimss_textzoom.sh
```

## Inference
To generate SR images from the LR images of TextZomm with a trained DiMSS, run the script via
```
bash eval_dimss_textzoom.sh
```
Also, use the following script to generate SR images of TextZoon with a trained GT-DiMSS
```
bash eval_gt_dimss_textzoom.sh
```

# LR-HR Paired Text Image Synthesis

## Training Synthesizer

### 1. Dataset

To train Synthesizer, the preprocessed STR dataset is required in addition to the TextZoom dataset.
Download the preprocessed STR dataset from
```
https://github.com/ku21fan/STR-Fewer-Labels
```

The structure of ``dataset`` directory is
```
dataset
`-- data_CVPR2021
    `-- training
        `-- label
            `-- real
                |-- 1.SVT
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 10.MLT19
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 11.ReCTS
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 2.IIIT
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 3.IC13
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 4.IC15
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 5.COCO
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 6.RCTW17
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 7.Uber
                |   |-- data.mdb
                |   `-- lock.mdb
                |-- 8.ArT
                |   |-- data.mdb
                |   `-- lock.mdb
                `-- 9.LSVT
                    |-- data.mdb
                    `-- lock.mdb
```

To perform the preprocessing for the Synthesizer training, run the script via
```
python preprocessing_STR.py
```
When the preprocessing is complete, preprossed text images and the corresponding text labels are placed in ```dataset/STR/img``` and ```dataset/STR/word```, respectively.

### 2. Training
To train Synthesizer, run the script via
```
bash train_synthesizer.sh
```

## Training Super-resolver
Super-resolver is identical to GT-DiMSS trained on TextZoom. 
See the DiMSS section described eariler.

## Training Degrader
Degrader is trained on TextZoom only. To train Degrader, run the script via:
```
bash train_degrader.sh
```

## Synthesizing Text Images

### 1. Synthesizer
To run Synthesizer, run the script via:
```
bash run_synthesizer.sh
```
The generated text images and the corresponding text labels are placed in ```./diff_samples/mr_samples```.

### 2. Postprocessing
To perform the preprocessing for the generated text images, run the script via:
```
python postprocessing_text_images.py
```
The postprocessed text images are placed in ```./diff_samples/mr_samples/postprocessed```.

## Generating LR and HR text images

### 1. Super-resolver
To run Super-resolver, run the script via:
```
bash run_super_resolver.sh
```
The generated HR text images are placed in  ```./diff_samples/hr_samples```.

### 2. Degrader
To run Degrader, run the script via:
```
bash run_degrader.sh
```
The generated LR text images are placed in  ```./diff_samples/lr_samples```.




## Citation
```
@article{noguchi2023scene,
  title={Scene Text Image Super-resolution based on Text-conditional Diffusion Models},
  author={Noguchi, Chihiro and Fukuda, Shun and Yamanaka, Masao},
  journal={arXiv preprint arXiv:2311.09759},
  year={2023}
}
```

## Licence
The code will be released with the MIT license.
