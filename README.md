<h1 align="center"><strong>ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment</strong></h1>
<p align="center">
 <a href='https://github.com/wengwanjiang' target='_blank'>Wanjiang Weng<sup>*</sup></a>&emsp;
 <a href='https://xiaofeng-tan.github.io/' target='_blank'>Xiaofeng Tan<sup>*</sup></a>&emsp;
 Junbo Wang&emsp;
 Guo-Sen Xie&emsp;
 Pan Zhou&emsp;
 Hongsong Wang<sup>†</sup>&emsp;
  <br>
  *Equal Contribution&emsp;
  †Corresponding Author
</p>

<p align="center">
  <a href="https://aaai.org/conference/aaai/aaai-26/">
    <img src="https://img.shields.io/badge/AAAI-2026-138D75" alt="AAAI 2026">
  </a>
  <a href="https://arxiv.org/abs/2511.19217">
    <img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper PDF on arXiv">
  </a>
 <a href='https://wengwanjiang.github.io/ReAlign-page'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
</p>

> **TL;DR:** We propose **ReAlign**, a *plug-and-play reward-guided alignment strategy* for text-to-motion generation, which explicitly enhances both semantic consistency and motion realism throughout the denoising process.

This repository offers the official code for this paper. If you have any questions, feel free to contact Wanjiang Weng (wjweng@seu.edu.cn) or Xiaofeng Tan (xiaofengtan@seu.edu.cn).

## 📣 News
- **[2025/12]** The code has been released! 🚀
- **[2025/11]** The paper has been publicly released.
- **[2025/11]** 🎉 **ReAlign** has been accepted by **AAAI‘26**. To explore accepted papers from AAAI’26, please see  [AAAI Abstract](https://hongsong-wang.github.io/AAAI2026_Abstract/) and [Paper Portal](https://hongsong-wang.github.io/CV_Paper_Portal/).
  
## 📆 Plan
- [x] Release early version.
- [x] Release [final version](https://arxiv.org/abs/2511.19217).
- [x] Release environment guidance.
- [x] Release evaluation code.
- [x] Release training code.
- [x] Release pretrained model weights.



## Model Zoo
<table>
  <tr>
    <th>Model Name</th>
    <th>Dataset</th>
    <th>Download Link</th>
    <th>Retrieval Performance (R@1)</th>
  </tr>
  <tr>
    <td rowspan="2">Step-Aware Reward Model</td>
    <td>HumanML3D</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a>,
      <a href="https://pan.baidu.com/s/1HHux8t_cCaENw9_ybrGIOg">BaiduNetDisk (passwd: 1234)</a>
    </td>
    <td>T2M: 67.59%, M2T: 68.94%</td>
  </tr>
  <tr>
    <td>KIT-ML</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a>,
      <a href="https://pan.baidu.com/s/1HHux8t_cCaENw9_ybrGIOg">BaiduNetDisk (passwd: 1234)</a>
    </td>
    <td>T2M: 52.84%, M2T: 52.98%</td>
  </tr>
</table>

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n realign python=3.10 -y
conda activate realign
```

### 2. Install Dependencies

```bash
# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Prepare Pre-trained Models

Download and place the following models in the `deps/` directory:

```
deps/
├── sentence-t5-large/      # Sentence-T5 for text encoding
├── clip-vit-large-patch14/ # CLIP model
├── distilbert-base-uncased/
├── glove/                  # GloVe word embeddings
├── smpl/                   # SMPL model
└── t2m/                    # Text-to-Motion evaluation model
```

### 4. Prepare Dataset

Download HumanML3D or KIT-ML dataset and place in `datasets/` directory:

```
datasets/
└── humanml3d/
    ├── new_joint_vecs/
    ├── new_joints/
    ├── texts/
    └── ...
```

## SwanLab Logging

This project uses [SwanLab](https://swanlab.cn/) for experiment tracking and visualization.

### Setup SwanLab

```bash
# Install SwanLab
pip install swanlab

# Login to SwanLab (get API key from https://swanlab.cn/)
swanlab login
```

### View Training Logs

After training starts, you can view real-time logs at:
- **SwanLab Dashboard**: https://swanlab.cn/

Logged metrics include:
- `Train/loss`: Training loss per step
- `Train/lr`: Learning rate
- `Epoch/avg_loss`: Average loss per epoch
- `Val/T2M_*`: Text-to-Motion retrieval metrics
- `Val/M2T_*`: Motion-to-Text retrieval metrics

## Training

### Train Step-Aware Reward Model (SPM)

```bash
# Train on HumanML3D dataset
bash run.sh 0 spm

# Or run directly with custom parameters
CUDA_VISIBLE_DEVICES=0 python -m ReAlignModule.train_spm \
    --cfg configs/spm_t2m.yaml \
    --NoiseThr 0.5 \
    --maxT 1000 \
    --step_aware M1T0 \
    --CLThr 0.9 \
    --CLTemp 0.1
```

**Key Parameters:**
- `--NoiseThr`: Noise threshold for training (default: 0.5)
- `--maxT`: Maximum timestep for noise scheduling (default: 1000)
- `--step_aware`: Step-aware mode, options: `M1T0`, `M0T1`, `M1T1` (default: M1T0)
- `--CLThr`: Contrastive learning threshold (default: 0.9)
- `--CLTemp`: Contrastive learning temperature (default: 0.1)

Checkpoints will be saved in `./checkpoints/spm/` directory.

## Evaluation

### Evaluate with ReAlign

```bash
# Evaluate MLD with Step-Aware Reward Model
bash run.sh 0 eval

# Or run directly
CUDA_VISIBLE_DEVICES=0 python -m test \
    --cfg configs/mld_t2m.yaml \
    --lambda_t2m 100 \
    --lambda_m2m 100 \
    --spm_path /path/to/SPM_checkpoint.pth
```

**Key Parameters:**
- `--spm_path`: Path to trained SPM checkpoint
- `--lambda_t2m`: Weight for text-to-motion alignment (default: 100)
- `--lambda_m2m`: Weight for motion-to-motion alignment (default: 100)

### Evaluate TMR Retrieval

```bash
bash run.sh 0 tmr
```

## Rendering

Convert generated motion to mesh for visualization:

```bash
python fit.py --pkl /path/to/motion.pkl --num_smplify_iters 150
```

## Project Structure

```
ReAlign/
├── configs/                 # Configuration files
│   ├── mld_t2m.yaml        # MLD model config
│   ├── spm_t2m.yaml        # SPM training config
│   └── modules/            # Module-specific configs
├── ReAlignModule/          # ReAlign core module
│   ├── train_spm.py        # SPM training script
│   ├── eval_tmr.py         # TMR evaluation
│   └── models/
│       ├── spm.py          # Step-aware Reward Model
│       └── utils.py        # Utility functions
├── mld/                    # MLD base model
├── datasets/               # Dataset directory
├── deps/                   # Pre-trained dependencies
├── checkpoints/            # Model checkpoints
├── run.sh                  # Training/evaluation scripts
├── test.py                 # Evaluation entry point
└── fit.py                  # Motion to mesh fitting
```

## Update Log

### 2025-12-15
- **Bug Fix**: Fixed GPU memory leak issue in `train_spm.py`
  - Added `del` statements to explicitly release tensors after use
  - Added `torch.cuda.empty_cache()` calls to free cached memory
  - Prevents OOM errors during long training sessions

## Citation

```bibtex
@inproceedings{wengReAlign26,
  title={ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment}, 
  author={Wanjiang Weng and Xiaofeng Tan and Junbo Wang and Guo-Sen Xie and Pan Zhou and Hongsong Wang},
  year={2025},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}
}
```
