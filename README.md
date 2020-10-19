[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](./LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
<span id="jump1"></span>
# Rethinking-Inpainting-MEDFE
![MEDFE Show](./show.png)

###  [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470715.pdf) | [BibTex](#jump2)

Rethinking Image Inpainting via a Mutual Encoder Decoder with Feature Equalizations .<br>

[Hongyu Liu](#jump1),  [Bin Jiang](#jump1), [Yibing Song](https://ybsong00.github.io/), [Wei Huang](#jump1) and [Chao Yang](#jump1).<br>
In ECCV 2020 (Oral).

### [License](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)


All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [kumapower@hnu.edu.cn](#jump1).


## Installation

Clone this repo.
```bash
git clone https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE.git
```

Prerequisites
* Python3
* Pytorch >=1.0
* Tensorboard
* Torchvision
* pillow


## Dataset Preparation

We use [Places2](http://places2.csail.mit.edu/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites.

Our model is trained on the irregular mask dataset provided by [Liu et al](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from their [website](http://masc.cs.gmu.edu/wiki/partialconv).


For Structure image of datasets, we follow the [structure flow](https://github.com/RenYurui/StructureFlow) and utlize the [RTV smooth method](http://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/).Run generation function [data/Matlab/generate_structre_images.m](./data/Matlab/generate_structure_images.m) in your matlab. For example, if you want to generate smooth images for Places2, you can run the following code:
```
generate_structure_images("path to Places2 dataset root", "path to output folder");
```


## Training New Models
```bash
# To train on the you dataset, for example.
python train.py --st_root=[the path of structure images] --de_root=[the path of ground truth images] --mask_root=[the path of mask images]
```
There are many options you can specify. Please use `python train.py --help` or see the options

For the current version, the batchsize needs to be set to 1.

To log training, use `--./logs` for Tensorboard. The logs are stored at `logs/[name]`.



## Code Structure

- `train.py`: the entry point for training.
- `models/networks.py`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well.
- `data/`: process the dataset before passing to the network.
- `models/encoder.py`: defines the encoder.
- `models/decoder.py`: defines the decoder.
- `models/PCconv.py`: defines the Multiscale Partial Conv, feature equalizations and two branch.
- `models/MEDFE.py`: defines the loss, model, optimizetion, foward, backward and others.


## Pre-trained weights and test model
There are three folders to present pre-trained for three datasets respectively, for the celeba, we olny use the centering masks. You can download the pre-trained model [here](https://drive.google.com/drive/folders/1uLC9YN_34mLod5kIE1nMb9P5L40Iqbkp?usp=sharing). The test model and demo will comming soon. I think adding random noise to the input may make the effect better， I will re-train our model and update the parameters.

## About Feature equalizations
I think the feature equalizations may can be utlized in many tasks to replace the traditional attention block (None local/CBAM). I didn't try because of lack of time，I hope someone can try the method and communicate with me.


<span id="jump2"></span>
### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{Liu2019MEDFE,
  title={Rethinking Image Inpainting via a Mutual Encoder-Decoder with Feature Equalizations},
  author={Hongyu Liu, Bin Jiang, Yibing Song, Wei Huang, and Chao Yang,},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
```


