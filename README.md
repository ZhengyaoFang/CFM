# Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://zhengyaofang.github.io/CFM/)
[![CVPR](https://img.shields.io/badge/CVPR-2026-8c1b13.svg)](https://openaccess.thecvf.com/content/CVPR2026/papers/Fang_Too_Vivid_to_Be_Real_Benchmarking_and_Calibrating_Generative_Color_CVPR_2026_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2603.10990-b31b1b.svg)](https://arxiv.org/abs/2603.10990)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Nineve/CFM_7B)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/Nineve/CFD_dataset)



This repository contains the official implementation of **"Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity"**, which has been **accepted to CVPR 2026**.

## 📢 News
- **2026-04**: Our paper is selected as a Highlight at CVPR 2026.
- **2026-03**: Our paper *"Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity"* is **accepted to CVPR 2026**.

## 📄 Paper

- **Title**: Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity  
- **Conference**: CVPR 2026  
- **Project page**: [https://zhengyaofang.github.io/CFM/](https://zhengyaofang.github.io/CFM/)
- **CVPR 2026 paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/papers/Fang_Too_Vivid_to_Be_Real_Benchmarking_and_Calibrating_Generative_Color_CVPR_2026_paper.pdf)
- **Paper link**: [arXiv:2603.10990](https://arxiv.org/abs/2603.10990)

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
@InProceedings{Fang_2026_CVPR,
    author    = {Fang, Zhengyao and Jia, Zexi and Zhong, Yijia and Luo, Pengcheng and Zhang, Jinchao and Lu, Guangming and Yu, Jun and Pei, Wenjie},
    title     = {Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37258-37267}
}
```

We will update the BibTeX entry with the final author list and publication details once they are available.

## 📜 License

This project is licensed under the **Apache License 2.0**. 
