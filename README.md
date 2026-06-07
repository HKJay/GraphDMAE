# GraphDMAE

Official code for the paper **"GraphDMAE: A Novel Approach for Graph Adversarial Defense via Spectral Stabilization"**.

## Overview

GraphDMAE is a robust graph neural network defense framework that **converts adversarial structural perturbations into controllable random sparse perturbations** by re‑establishing the Davis–Kahan spectral stability condition. The framework consists of three stages:

1. **Skeleton Graph Construction**: constructs Skeleton Graph and extracts Laplacian structural features.
2. **Denoising Masked Autoencoder**: jointly compresses the perturbation norm and amplifies the spectral gap in latent space.
3. **Spectral Reconstruction & Secondary Filtering**: produces a stabilized graph and robust node embeddings for downstream classification.

![GraphDMAE Framework](./figures/framework1.png)

## Quick Start

### Environment

- python==3.13.12
- deeprobust==0.2.11
- matplotlib==3.10.9
- numpy==2.4.6
- pandas==3.0.3
- scikit\_learn==1.8.0
- scipy==1.17.1
- torch==2.6.0+cu124
- torch\_geometric==2.7.0

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset & Attack Preparation

We evaluate on four datasets: **Cora, Citeseer, Cora‑ML, Pubmed**. Dataset statistics are shown in the table below. We only consider the largest connected component.

| Dataset  | Nodes | Edges | Classes | Features |
| -------- | ----- | ----- | ------- | -------- |
| Cora     | 2485  | 5069  | 7       | 1433     |
| Citeseer | 2110  | 3668  | 6       | 3703     |
| Cora‑ML  | 2810  | 7981  | 7       | 2879     |
| Pubmed   | 19717 | 44338 | 3       | 500      |

The attacked graphs are generated using [DeepRobust](https://github.com/DSE-MSU/DeepRobust). We provide pre‑computed perturbed graphs for reproducibility.
You can sample the attacked graphs by running the following python script:

```bash
python structure_attack.py --dataset cora --attack mettack --ptb_rate 0.25
```

Or download the precomputed attacked graphs (Only for Cora and Citeseer under Mettack and Nettack, other datasets are not provided):

```bash
python load_data.py --dataset cora --attack mettack --ptb_rate 0.25
```

The data will be placed in `./ptb_graphs/` with the following structure:

```
ptb_graphs/
├── cora_features.npz
├── cora_labels.npy
├── ...
├── mettack/
│   ├── mettack_cora_0.05.pt
│   ├── mettack_cora_0.05_idx_test.npy
│   ├── mettack_cora_0.05_idx_train.npy
│   ├── mettack_cora_0.05_idx_val.npy
│   └── ...
├── DICE/
├── nettack/
└── random/
```

### Running GraphDMAE

To run the full defense pipeline on Cora against Mettack (25% perturbation):

```bash
python main.py --use_config --config cora_mettack_0.25 --log
```

The script performs all three stages, then evaluates the downstream GAT classifier. Results are saved in `./log/`.

## Code Structure

```
GraphDMAE/
├── main.py                 # Entry point: runs the full pipeline
├── configs.json            # JSON configuration file
├── load_attacked.py        # Download attacked graphs
├── structure_attack.py     # Sample attacked graphs by structure perturbation
├── models/
│   ├── GraphDMAE.py        # Stage 2: Graph Denoising Masked Autoencoder
├── utils/
│   └── utils.py            # Utility functions(including stages 1, 3)
├── ptb_graphs/             # attacked graphs
│   └── ...
├── tmp/                    # temporary files
├── log/                    # log files
│   └── mettack/
│       └── ...
│   └── DICE/
│       └── ...
│   └── nettack/
│       └── ...
│   └── random/
│       └── ...
├── figures/                # figures
├── results.xlsx            # results of experiments
├── requirements.txt
└── README.md
```

## Hyperparameters

Key hyperparameters and their default values:

| Parameter    | Description                                 | Default |
| ------------ | ------------------------------------------- | ------- |
| `k`          | Number of Laplacian eigenvectors            | 20      |
| `p_mask`\*   | Masking ratio for cross‑masking             | 0.5     |
| `p_swap`\*   | Feature swapping ratio                      | 0.1     |
| `γ`\*        | Scaling exponent for cosine loss            | 2.0     |
| `τ_l`\*      | Similarity threshold in adaptive smoothness | 0.8     |
| `τ_c`        | Feature similarity threshold (skeleton)     | 0.6     |
| `τ_j`        | Structural similarity threshold (skeleton)  | 0.6     |
| `τ_add`      | Edge supplementation threshold              | 0.85    |
| `k_l`        | Neighbors for contrastive loss              | 50      |
| `τ_lap`\*    | Temperature in contrastive loss             | 1.0     |
| `τ_re`       | Spectral reconstruction threshold           | 0.4     |
| `τ_sec`      | Secondary filtering threshold               | 0.6     |
| `α`\*        | Weight for Laplacian corrector loss         | 0.5     |
| `β`\*        | Weight for smoothness loss                  | 0.1     |

\* denotes the value used in all the datasets.
More details about the hyperparameters can be found in config.json.

## Results

Output results of main experiments can be found in ./log/.
To maintain reproducibility, all experimental records are retained as original. The variable naming in source code and log files follows engineering conventions during early development. The following table provides a one-to-one correspondence between parameter in the paper, source code variables, and logged fields.

| Parameter | Field Name in Log and code |
| --------- | -------------------------- |
| `k`       | `L_dim`                    |
| `τ_c`     | `cos`                      |
| `τ_j`     | `jt`                       |
| `τ_add`   | `cos_add`                  |
| `k_l`     | `k`                        |
| `τ_re`    | `lap_threshold`            |
| `τ_sec`   | `recover_threshold`        |

The other results are saved in **results.xlsx**.

### Average Running Time per Dataset

The running time is measured as the duration between the first and last log entry in each `.log` file under `./log/`.

| Dataset  | Avg Time (s) | Min (s) | Max (s) |
|----------|:------------:|:-------:|:-------:|
| cora     |    43.33     |  24.00  |  58.00  |
| cora_ml  |    29.29     |  20.00  |  49.00  |
| citeseer |    42.83     |  32.00  |  58.00  |
| pubmed   |    70.14     |  67.00  |  74.00  |
