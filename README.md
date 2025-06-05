# CADFormer

## Setting Up

The code has been verified to work with PyTorch v1.8.1 and Python 3.7，other versions may also be compatible.

1. Follow [RMSIN](https://github.com/Lsan2401/RMSIN) instructions for environment set-up.
2. Download Initialization [pretrained weights](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [BERT weights](https://huggingface.co/google-bert/bert-base-uncased).

## Datasets

We build the RRSIS-HR dataset based on the [RSVG-HR](https://github.com/LANMNG/LQVG) dataset. It can be downloaded from !!!. The RRSIS-D dataset could be download from [Here](https://drive.google.com/drive/folders/1Xqi3Am2Vgm4a5tHqiV9tfaqKNovcuK3A).

## Acknowledgements

Code in this repository is built on [RMSIN](https://github.com/Lsan2401/RMSIN) and [LAVT](https://github.com/yz93/LAVT-RIS). We'd like to thank the authors for open sourcing their project.