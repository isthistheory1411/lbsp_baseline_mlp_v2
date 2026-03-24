# MLP (v.2) LBSP
This repository contains the MLP_V2 model for predicting ligand binding residues on proteins from per-residue embeddings. The model is HPC-ready, supports training with early stopping, per-residue inference, and configurable evaluation metrics.

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Inference](#inference)
7. [Reproducibility](#reproducibility)
8. [Output Files](#output-files)
9. [Example Usage](#example-usage)

## Overview
The MLP_V2 model predicts the likelihood of each residue in a protein being part of a ligand binding site. It uses pre-computed per-residue embeddings stored in HDF5 files and supports:

1. HPC-ready training with mixed precision (`torch.cuda.amp`)
2. Early stopping and class imbalance weighting
3. Threshold optimization on validation sets
4. Per-residue inference with CSV output
5. Optional evaluation metrics if binding labels are available

## Directory Structure
```
MLP_V2/
│
├─ src/
│   ├─ dataset.py        # Dataset and DataLoader helpers
│   ├─ model.py          # ResidueMLP model definition
│   ├─ train.py          # Training loop
│   ├─ evaluate.py       # Evaluation functions
│   ├─ utils.py          # Utility functions (e.g., set_seed)
│   └─ main.py           # Training entry point
│
├─ inference/
│   ├─ inference.py      # Inference helper function
│   └─ inference_main.py # CLI entry point for inference
│
├─ config/
│   ├─ config.yaml             # Training configuration
│   └─ inference_config.yaml   # Inference configuration
│
├─ generate_embed/             # Example per-residue embeddings HDF5
├─ model_dev/                  # Example DataFrames and checkpoints
├─ README.md
└─ LICENSE
```

## Installation 
1. Clone the repository:
```
git clone https://github.com/yourusername/lbsp_baseline_mlp_v2.git
cd lbsp_baseline_mlp_v2
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Then install dependencies:
```
pip install -r mlp_v2_requirements.txt
```

## Configuration
Training and Inference are fully configurable using YAML files:
- `config/config.yaml` – Training dataset paths, model hyperparameters, optimizer settings, and checkpoint locations.
- `config/inference_config.yaml` – Test dataset, checkpoint, threshold, batch size, and device for inference.

Example:
```
model:
  residue_emb_dim: 1024
  protein_emb_dim: 256
  hidden_dims: [512, 256, 128]
  dropout: 0.1
  max_len: 1022
```

## Training
Run the full HPC-ready training pipeline:
```
python src/main.py --config config/config.yaml
```
The training pipeline includes:

- Early stopping with patience
- Optimal threshold selection for MCC
- Saving the best model checkpoint
- Storing training / validation loss histories

## Inference
Before running Inference mode, update `config/inference_config.yaml`:
- `path.checkpoint` -> path to trained model
- `data.test_df` -> path to test dataframe
- `data.h5_embeddings` -> path to HDF5 embeddings
- `path.inference.csv` -> output location

Run per-residue predictions with optional metrics if labels are available. The full YAML path is recommended:
```
export PYTHONPATH=$(pwd) python -m inference.inference_main --config full/path/to/config/inference_config.yaml
```

Outputs:
- CSV file (i.e. `inference_results.csv`) containing:
  - `protein_key`
  - `residue_index`
  - `probability`
  - `prediction` (binary)
- **JSON metrics file** (if labels provided) containing: ROC-AUC, AU-PRC, MCC, Precision, Recall

You can override the threshold at runtime:
```
python inference_main.py --config config/inference_config.yaml --override inference.threshold=0.6
```

## Reproducibility
Random seeds are set using `src/utils.py`:
```
from src.utils import set_seed
set_seed(42)
```
Seeds are applied in both `main.py` (training) and `inference_main.py` (inference). This ensures consistent results across runs.

## Output Files
- `best_model.pt` – Saved model checkpoint
- `training_results.joblib` – Training/validation losses, threshold metrics
- `predictions.csv` – Per-residue predictions
- `predictions_metrics.json` – Evaluation metrics if available

## Example Usage
1. Train on example dataset:
```
python src/main.py --config config/config.yaml
```
  Download example per-residue [ProtT5 embeddings](https://drive.google.com/file/d/1t5mn4YDiVk_aVm_2GkjAN8G9AzBy7Mix/view?usp=drive_link)

2. Run Inference on example data:
```
export PYTHONPATH=$(pwd) python -m inference.inference_main --config full/path/to/config/inference_config.yaml
```
