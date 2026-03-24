import os
import torch
import joblib
import argparse
import numpy as np
from omegaconf import OmegaConf

# Import modules
from src.dataset import get_protein_dataloader
from src.model import ResidueMLP
from src.train import train_model_hpc
from src.evaluate import evaluate_on_test_hpc, find_optimal_threshold
from src.utils import set_seed, save_results

def execute_training_pipeline_hpc(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    pos_weight,
    device,
    num_epochs=50,
    patience=5,
    save_path=None,
    thresholds=None
):
    """
    Full HPC-ready training pipeline.
    """
    if save_path is None:
        raise ValueError("You must specify a full path for the model checkpoint (save_path).")

    # 1. Train model with early stopping
    train_loss_history, val_loss_history, _ = train_model_hpc(
        model, train_loader, val_loader, optimizer, pos_weight, device,
        num_epochs=num_epochs, patience=patience, save_path=save_path
    )

    # 2. Load best model checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    # 3. Find optimal threshold on validation set
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.1)
    optimal_threshold, threshold_results = find_optimal_threshold(
        model, val_loader, device, thresholds=thresholds
    )

    # 4. Evaluate on test set
    test_metrics = evaluate_on_test_hpc(
        model, test_loader, device, threshold=optimal_threshold
    )

    return test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results


if __name__ == "__main__":
    # -------------------------------
    # Command-line arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="Ligand Binding Site Prediction Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # -------------------------------
    # Load config using OmegaConf
    # -------------------------------
    cfg = OmegaConf.load(os.path.expanduser(args.config))
    if args.override:
        for override in args.override:
            key, value = override.split("=")
            OmegaConf.update(cfg, key, value)

    # -------------------------------
    # Set random seed for reproducibility
    # -------------------------------
    set_seed(cfg.training.seed)

    # -------------------------------
    # Paths
    # -------------------------------
    train_pkl = os.path.expanduser(cfg.data.train_df)
    val_pkl = os.path.expanduser(cfg.data.val_df)
    test_pkl = os.path.expanduser(cfg.data.test_df)
    h5_path = os.path.expanduser(cfg.data.h5_embeddings)
    checkpoint_path = os.path.expanduser(cfg.paths.checkpoint)
    results_path = os.path.expanduser(cfg.paths.results)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # -------------------------------
    # Load DataFrames
    # -------------------------------
    train_df = joblib.load(train_pkl)
    val_df = joblib.load(val_pkl)
    test_df = joblib.load(test_pkl)

    # -------------------------------
    # Create DataLoaders using helper
    # -------------------------------
    train_loader = get_protein_dataloader(
        train_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        max_len=cfg.model.max_len
    )
    val_loader = get_protein_dataloader(
        val_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len
    )
    test_loader = get_protein_dataloader(
        test_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len
    )

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Model
    # -------------------------------
    model = ResidueMLP(
        residue_emb_dim=cfg.model.residue_emb_dim,
        protein_emb_dim=cfg.model.protein_emb_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout
    )
    model.to(device)

    # -------------------------------
    # Optimizer and class imbalance
    # -------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    pos_weight = torch.tensor([cfg.training.pos_weight], device=device)

    # -------------------------------
    # Execute training pipeline
    # -------------------------------
    test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results = execute_training_pipeline_hpc(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        pos_weight=pos_weight,
        device=device,
        num_epochs=cfg.training.num_epochs,
        patience=cfg.training.patience,
        save_path=checkpoint_path,
        thresholds=cfg.get("evaluation", {}).get("thresholds", np.arange(0.5, 1.0, 0.1))
    )

    # -------------------------------
    # Save results
    # -------------------------------
    results_to_save = {
        "test_metrics": test_metrics,
        "optimal_threshold": optimal_threshold,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "threshold_results": threshold_results
    }
    save_results(results_to_save, save_path=results_path)
