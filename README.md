# CADFormer and Referring Remote Sensing Segmentation Benchmark (RRSIS-HR)

This repository contains the official PyTorch implementation of **[CADFormer: Fine-Grained Cross-Modal Alignment and Decoding Transformer for Referring Remote Sensing Image Segmentation](https://ieeexplore.ieee.org/abstract/document/11023843)**, accepted for publication in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)*.  

Along with CADFormer, we also release a **large-scale referring remote sensing segmentation benchmark**, **RRSIS-HR**, to advance research in cross-modal understanding for remote sensing imagery.



---

## ⚙️ Environment Setup

The codebase has been verified with **PyTorch 1.8.1** and **Python 3.7**, but other recent versions may also be compatible.

1. Follow the [RMSIN](https://github.com/Lsan2401/RMSIN) guidelines for environment setup.  
2. Download the following pretrained weights:  
   - [Swin-Transformer backbone](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)  
   - [BERT weights](https://huggingface.co/google-bert/bert-base-uncased)  

---

## 📂 Datasets

- **RRSIS-HR (Proposed Dataset)**  
  - [Baidu Netdisk](https://pan.baidu.com/s/1VG6sxkOuzeWelE7zkVv1Sw?pwd=wust)  
  - [OneDrive](https://onedrive.live.com/?ls=true&cid=9D2E24A3EBA4223F&id=9D2E24A3EBA4223F%21s9594f91279ad492d8d8ed1c3e8d628e5&parId=root&o=OneUp)  

- **Additional Benchmark**  
  - [RRSIS-D](https://drive.google.com/drive/folders/1Xqi3Am2Vgm4a5tHqiV9tfaqKNovcuK3A)  

---

## 🙏 Acknowledgements

This repository builds upon the excellent open-source contributions of:  
- [RMSIN](https://github.com/Lsan2401/RMSIN)  
- [LAVT](https://github.com/yz93/LAVT-RIS)  
The proposed **RRSIS-HR** dataset is constructed based on [RSVG-HR](https://github.com/LANMNG/LQVG). We sincerely thank the authors for their valuable efforts and open sharing of code and data.  

---

## 🔍 Highlights

- **CADFormer** introduces fine-grained cross-modal alignment and decoding strategies tailored for referring segmentation in remote sensing imagery.  
- **RRSIS-HR** provides the largest high-resolution benchmark to date for referring image segmentation in the remote sensing domain.  
- Designed to facilitate reproducibility, benchmarking, and future research in multimodal learning for Earth observation.  

## 📑 Citation
If you find this research or dataset useful for your research, please cite our paper:
```
@ARTICLE{11023843,
  author={Liu, Maofu and Jiang, Xin and Zhang, Xiaokang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={CADFormer: Fine-Grained Cross-Modal Alignment and Decoding Transformer for Referring Remote Sensing Image Segmentation}, 
  year={2025},
  volume={18},
  number={},
  pages={14557-14569},
  keywords={Remote sensing;Visualization;Image segmentation;Decoding;Semantics;Sports;Feature extraction;Transformers;Accuracy;Semantic segmentation;Cross-modal alignment;referring image segmentation (RIS);remote sensing},
  doi={10.1109/JSTARS.2025.3576595}}
```

