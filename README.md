# Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity

[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Nineve/CFM_7B)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/Nineve/CFD_dataset)

This repository contains the official implementation of **"Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity"**, which has been **accepted to CVPR 2026**.

## 📢 News

- **2026-03**: Our paper *"Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity"* is **accepted to CVPR 2026**.

## 📄 Paper

- **Title**: Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity  
- **Conference**: CVPR 2026  
- **Paper link**: *coming soon*

If you find this repository or the paper helpful in your research, please consider citing our work and starring this repository.

## 🔧 Installation

We recommend creating a dedicated conda environment:

```bash
conda create -n cfm python=3.11
conda activate cfm

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# (Optional) FlashAttention for efficiency
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Other dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Models and Weights

- **Hugging Face model** (CFM-7B weights and configs): [Nineve/CFM_7B](https://huggingface.co/Nineve/CFM_7B)

### Datasets

- **Color Fidelity Dataset (CFD)** on Hugging Face: [Nineve/CFD_dataset](https://huggingface.co/datasets/Nineve/CFD_dataset)


## 📚 Citation

If you use this project or find our paper useful in your research, please cite:

```bibtex
@inproceedings{cfm2026toovivid,
  title     = {Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity},
  author    = {Zhengyao Fang, Zexi Jia, Yijia Zhong, Pengcheng Luo, Jinchao Zhang, Guangming Lu, Jun Yu, Wenjie Pei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

We will update the BibTeX entry with the final author list and publication details once they are available.

## 📜 License

This project is licensed under the **Apache License 2.0**. 