# WaveSP-Net: Learnable Wavelet-Domain Sparse Prompt Tuning for Speech Deepfake Detection

[![arXiv](https://img.shields.io/badge/arXiv-2508.09294v1-b31b1b.svg)](https://arxiv.org/abs/2510.05305) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/xxuan-speech/WaveSP-Net)

## Getting Started

### 1 Setup Environment
You need to create the running environment by [Anaconda](https://www.anaconda.com/).

First, create and activate the environment:

```bash
conda create -n WaveSP-Net python=3.8
conda activate WaveSP-Net
```

Install fairseq:

```bash
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

### 2 Download Deepfake-Eval-2024 and SpoofCeleb Datasets

Our experiments are conducted on two new and challenging benchmarks:  

**Deepfake-Eval-2024** ([https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024)) 


**SpoofCeleb** ([https://huggingface.co/datasets/jungjee/spoofceleb](https://huggingface.co/datasets/jungjee/spoofceleb)).


### WaveSP-Net on Hugging Face 🤗

Our model is available on Hugging Face: [https://huggingface.co/xxuan-speech/WaveSP-Net](https://huggingface.co/xxuan-speech/WaveSP-Net)


### Citation

If you find our repository valuable for your work, please consider citing our paper:

```bibtex
@misc{xuan2025wavespnet,
      title={WaveSP-Net: Learnable Wavelet-Domain Sparse Prompt Tuning for Speech Deepfake Detection},
      author={Xi Xuan and Xuechen Liu and Wenxin Zhang and Yi-Cheng Lin and Xiaojian Lin and Tomi Kinnunen},
      year={2025},
      eprint={2510.05305},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={[https://arxiv.org/abs/2510.05305](https://arxiv.org/abs/2510.05305)},
}
