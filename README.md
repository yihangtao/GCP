# GCP: Guarded Collaborative Perception with Spatial-Temporal Aware Malicious Agent Detection

[Yihang Tao*](https://scholar.google.com/citations?user=xxxxx), [Senkang Hu*](https://scholar.google.com/citations?user=xxxxx), Yue Hu, Haonan An, Hangcheng Cao, and [Yuguang Fang](https://scholar.google.com/citations?user=xxxxx), Fellow, IEEE

Accepted by **IEEE Transactions on Dependable and Secure Computing (TDSC)**.

[**IEEE TDSC PDF**](https://ieeexplore.ieee.org/document/11523166) | [**ArXiv Paper**](https://arxiv.org/abs/2501.02450)

GCP is a spatial-temporal defense framework for robust collaborative perception. This repository contains the core runtime for GCP, the BAC attack implementation, and the LSTM-AE training and inference code used for temporal anomaly detection.

## Installation

### Requirements

- Linux
- Python 3.7+
- Anaconda
- PyTorch 1.12+
- CUDA 11.7

### Environment

```bash
cd coperception
conda env create -f environment.yml
conda activate coperception
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .
```

## Data And Checkpoints

### V2X-Sim

Download the [V2X-Sim detection dataset](https://drive.google.com/file/d/1ZM_JkugZHmTwkR1gwG8ZuFq0YBwPDcDV/view?usp=drive_link) and extract it under the standard `train/` and `test/` split.

### Detection Checkpoints

Download [pre-trained detection weights](https://drive.google.com/drive/folders/1dGEYIzc5ITFKR0TSZfXPYAIw2GBo4oBT?usp=share_link) and save them in `coperception/ckpt/`:

- `meanfusion/epoch_49.pth`
- `meanfusion/epoch_advtrain_49.pth`

### LSTM-AE Checkpoint

Download the pre-trained `LSTM-AE` checkpoint from [Google Drive](https://drive.google.com/file/d/1hm-StGJD2dmNLd1zbVrAsLaJoUop56Yx/view?usp=sharing).

- Window length: `F = 5`
- Save path: `coperception/logs/model/best_model.pth`

### BEVFlow Dataset

`LSTM-AE` training uses a scene-level `.npz` file such as `coperception/logs/scene_0_ego_1.npz`.

- Generation logic: `coperception/coperception/datasets/BEVFlowGeneration.py`
- One file corresponds to one fixed `scene_id` and `ego_agent`
- Saved fields include `bev_flows`, `frame_ids`, `ego_id`, and `scene_id`
- Training samples are built in `train_lstm_ae.py` by sliding windows and Hungarian + IoU matching

## Quick Start

### Train LSTM-AE

```bash
cd /path/to/GCP
./scripts/train_lstm_ae.sh \
    --data_path ./coperception/logs/scene_0_ego_1.npz \
    --save_path ./coperception/logs/model/ \
    --seq_length 5
```

### Run GCP

Default script behavior:

- Defense mode: `gcp`
- Attack mode: `pgd`
- Number of attackers: `1`
- Scene: `8`
- LSTM-AE checkpoint: `coperception/logs/model/best_model.pth`
- LSTM-AE window length: `F = 5`

```bash
cd /path/to/GCP
./scripts/run_gcp.sh
```

## Common Commands

```bash
# GCP with default settings
./scripts/run_gcp.sh

# GCP on a frame range
./scripts/run_gcp.sh gcp --scene_id 8 --sample_start 0 --sample_end 10

# No defense baseline
./scripts/run_gcp.sh no_defense --adv_method pgd --eps 0.5

# ROBOSAC baseline
./scripts/run_gcp.sh robosac --adv_method bac --step_budget 5

# GCP with BAC attack
./scripts/run_gcp.sh gcp --adv_method bac
```

Useful options:

- `--gcp`: `gcp`, `robosac_mAP`, `made`, `no_defense`, `upperbound`, `lowerbound`
- `--adv_method`: `pgd`, `bim`, `cw`, `simple_bac`, `bac`
- `--scene_id`: target scene id, default `8`
- `--sample_start`, `--sample_end`: inclusive frame range
- `--history_length`: temporal window length, default `5`
- `--reconstruction_threshold`: LSTM-AE anomaly threshold
- `--box_matching_thresh`: spatial matching threshold

## Project Structure

```text
GCP/
├── coperception/
│   ├── coperception/models/det/LSTMAutoencoder.py
│   ├── coperception/datasets/BEVFlowGeneration.py
│   ├── coperception/utils/bac_attack.py
│   └── tools/det/
│       ├── gcp.py
│       ├── train_lstm_ae.py
│       └── box_matching.py
├── scripts/
│   ├── run_gcp.sh
│   └── train_lstm_ae.sh
└── README.md
```

Core files:

- `coperception/tools/det/gcp.py`: GCP runtime, spatial consistency, temporal reconstruction, and defense logic
- `coperception/coperception/utils/bac_attack.py`: BAC attack implementation
- `coperception/coperception/models/det/LSTMAutoencoder.py`: shared LSTM-AE model for training and inference
- `coperception/tools/det/train_lstm_ae.py`: LSTM-AE training entrypoint

## Citation

```bibtex
@ARTICLE{11523166,
  author={Tao, Yihang and Hu, Senkang and Hu, Yue and An, Haonan and Cao, Hangcheng and Fang, Yuguang},
  journal={IEEE Transactions on Dependable and Secure Computing},
  title={GCP: Guarded Collaborative Perception with Spatial-Temporal Aware Malicious Agent Detection},
  year={2026},
  pages={1-14},
  doi={10.1109/TDSC.2026.3693684}
}
```

## Acknowledgment

This project builds upon [coperception](https://github.com/coperception/coperception) and incorporates ideas from [ROBOSAC](https://github.com/coperception/ROBOSAC) and [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
