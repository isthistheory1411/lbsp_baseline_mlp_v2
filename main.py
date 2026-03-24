def execute_training_pipeline_hpc(model, train_loader, val_loader, test_loader,
                                  optimizer, pos_weight, device,
                                  num_epochs=50, patience=5,
                                  save_path=None):
    """
    Full HPC-ready training pipeline.
    save_path: full path to save best model checkpoint
    """
    if save_path is None:
        raise ValueError("You must specify a full path for the model checkpoint (save_path).")

    # 1. Train model with early stopping
    train_loss_history, val_loss_history = train_model_hpc(
        model, train_loader, val_loader, optimizer, pos_weight, device,
        num_epochs=num_epochs, patience=patience, save_path=save_path
    )

    # 2. Load best model checkpoint
    model.load_state_dict(torch.load(save_path))
    model.to(device)

    # 3. Find optimal threshold on validation set
    thresholds = np.arange(0.5, 1.0, 0.1)
    optimal_threshold, threshold_results = find_optimal_threshold(
        model, val_loader, device, thresholds=thresholds
    )

    # 4. Evaluate on test set
    test_metrics = evaluate_on_test_hpc(
        model, test_loader, device, threshold=optimal_threshold
    )

    return test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results


def save_results(results, save_path):
    import joblib
    import os
    joblib.dump(results, save_path, compress=3)
    print(f"Training results saved to {save_path}")


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    import torch
    import joblib

    # -------------------------------
    # Paths and data
    # -------------------------------
    h5_path = os.path.expanduser("~/generate_embed/per_residue_embeddings.h5")
    train_pkl = os.path.expanduser("~/model_dev/ligysis_id30_train_df.pkl")
    val_pkl = os.path.expanduser("~/model_dev/ligysis_id30_val_df.pkl")
    test_pkl = os.path.expanduser("~/model_dev/ligysis_id30_test_df.pkl")

    # -------------------------------
    # Load DataFrames
    # -------------------------------
    train_df = joblib.load(train_pkl)
    val_df = joblib.load(val_pkl)
    test_df = joblib.load(test_pkl)

    # -------------------------------
    # Create datasets
    # -------------------------------
    train_dataset = ProteinDataset(train_df, h5_path)
    val_dataset = ProteinDataset(val_df, h5_path)
    test_dataset = ProteinDataset(test_df, h5_path)

    # -------------------------------
    # DataLoaders
    # -------------------------------
    batch_size = 64
    num_workers = 0  # start with 0; increase to 4+ when safe with HPC

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    # -------------------------------
    # Device and model
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_res = 1024
    protein_emb_dim = 256
    input_dim = D_res + protein_emb_dim + 1  # residue + protein + position

    model = ResidueMLP(
        residue_emb_dim=D_res,
        protein_emb_dim=protein_emb_dim,
        hidden_dims=[512, 256, 128],
        dropout=0.1
    )
    model.to(device)

    # -------------------------------
    # Optimizer and class imbalance
    # -------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_frac = 1 / 10
    pos_weight = torch.tensor([(1 - pos_frac) / pos_frac]).to(device)

    # -------------------------------
    # Train, find optimal threshold, evaluate
    # -------------------------------

    checkpoint_dir = os.path.dirname(os.path.expanduser("~/model_dev/baseline_mlp/best_model.pt"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results = execute_training_pipeline_hpc(
        model, train_loader, val_loader, test_loader,
        optimizer, pos_weight, device,
        num_epochs=50,
        patience=5,
        save_path=os.path.expanduser("~/model_dev/baseline_mlp/best_model.pt")
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

    results_dir = os.path.expanduser("~/model_dev/baseline_mlp")  # e.g., your HPC scratch space
    os.makedirs(results_dir, exist_ok=True)  # ensure the folder exists
    save_path = os.path.join(results_dir, "training_results.joblib")

    save_results(results_to_save, save_path=save_path)
