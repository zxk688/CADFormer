# CADFormer

This is a repository for releasing a PyTorch implementation of [our work](https://ieeexplore.ieee.org/abstract/document/11023843) that has been accepted by IEEE JSTARS with a large-scale dataset named RRSIS-HR.

If you find this research or dataset useful for your research, please cite our paper:
```
@ARTICLE{11039674,
  author={Liu, Maofu and Liu, Jiahui and Zhang, Xiaokang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Semantic-Spatial Feature Fusion with Dynamic Graph Refinement for Remote Sensing Image Captioning}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Feature extraction;Visualization;Semantics;Remote sensing;Attention mechanisms;Decoding;Transformers;Vectors;Sensors;Convolutional neural networks;Remote sensing image captioning;feature fusion;graph attention;dynamic weighting mechanism},
  doi={10.1109/JSTARS.2025.3580686}}
```


## Setting Up

The code has been verified to work with PyTorch v1.8.1 and Python 3.7, other versions may also be compatible.

1. Follow [RMSIN](https://github.com/Lsan2401/RMSIN) instructions for environment set-up.
2. Download [pretrained weights](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [BERT weights](https://huggingface.co/google-bert/bert-base-uncased).

## Datasets

We created the RRSIS-HR dataset based on the [RSVG-HR](https://github.com/LANMNG/LQVG) dataset. RRSIS-HR can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1VG6sxkOuzeWelE7zkVv1Sw?pwd=wust) or [Google Drive].
Another dataset used in this study is [RRSIS-D](https://drive.google.com/drive/folders/1Xqi3Am2Vgm4a5tHqiV9tfaqKNovcuK3A).

## Acknowledgements

The codes are heavily borrowed from [RMSIN](https://github.com/Lsan2401/RMSIN) and [LAVT](https://github.com/yz93/LAVT-RIS). We'd like to thank the authors for their excellent work.
