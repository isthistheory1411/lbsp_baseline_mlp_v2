# -------------------------------
# Imports
# -------------------------------
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score
import joblib

# -------------------------------
# Dataset
# -------------------------------
class ProteinDataset(Dataset):
    def __init__(self, df, h5_path, max_len=1022):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_key = row['dataset_key']
        labels = row['binding_vector']

        with h5py.File(self.h5_path, 'r') as h5f:
            emb = h5f[protein_key][:]

        L_i, D = emb.shape
        if L_i < self.max_len:
            pad_len = self.max_len - L_i
            emb = np.vstack([emb, np.zeros((pad_len, D))])
            labels = np.hstack([labels, np.zeros(pad_len)])
        else:
            emb = emb[:self.max_len]
            labels = labels[:self.max_len]

        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:min(L_i, self.max_len)] = 1.0
        pos = np.arange(self.max_len) / self.max_len

        return {
            "embeddings": torch.tensor(emb, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "position": torch.tensor(pos, dtype=torch.float32)
        }

# Collate function
def collate_fn(batch):
    """
    batch: list of dicts from ProteinDataset
    Returns batched tensors ready for model input.
    """
    B = len(batch)
    L = batch[0]['embeddings'].shape[0]

    # Stack embeddings, labels, masks, positions
    emb = torch.stack([b['embeddings'] for b in batch], dim=0)   # [B,L,D_res]
    labels = torch.stack([b['labels'] for b in batch], dim=0)    # [B,L]
    mask = torch.stack([b['mask'] for b in batch], dim=0)        # [B,L]
    pos = torch.stack([b['position'] for b in batch], dim=0).unsqueeze(-1)  # [B,L,1]

    return {
        "embeddings": emb,  # [B,L,D_res]
        "labels": labels,   # [B,L]
        "mask": mask,       # [B,L]
        "position": pos     # [B,L,1]
    }


# -------------------------------
# Model
# -------------------------------
class ResidueMLP(nn.Module):
    def __init__(self, residue_emb_dim=1024, protein_emb_dim=256, hidden_dims=[512, 256, 128], dropout=0.1):
        """
        HPC-ready per-residue MLP with LayerNorm and dropout.

        Args:
            residue_emb_dim: dimension of per-residue embeddings (D_res)
            protein_emb_dim: projected per-protein embedding dimension
            hidden_dims: list of hidden layer sizes
            dropout: dropout probability
        """
        super().__init__()
        self.protein_proj = nn.Linear(residue_emb_dim, protein_emb_dim)

        # Total input dimension: residue + protein projection + position
        input_dim = residue_emb_dim + protein_emb_dim + 1
        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h

        layers.append(nn.Linear(last_dim, 1))  # per-residue logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings, mask, position):
        """
        embeddings: [B,L,D_res] per-residue embeddings
        mask: [B,L] valid residue mask
        position: [B,L,1] normalized residue positions

        returns: [B,L] logits per residue
        """
        # Compute per-protein mean embedding over valid residues
        mask_exp = mask.unsqueeze(-1)  # [B,L,1]
        sum_emb = (embeddings * mask_exp).sum(dim=1)  # [B,D_res]
        lengths = mask.sum(dim=1, keepdim=True)  # [B,1]
        mean_emb = sum_emb / lengths  # [B,D_res]

        # Linear projection to smaller per-protein embedding
        protein_feats = self.protein_proj(mean_emb)  # [B, protein_emb_dim]

        # Expand per-protein embedding to sequence length
        protein_feats_exp = protein_feats.unsqueeze(1).expand(-1, embeddings.size(1), -1)  # [B,L,protein_emb_dim]

        # Concatenate per-residue embedding + projected protein embedding + position
        x = torch.cat([embeddings, protein_feats_exp, position], dim=-1)  # [B,L,D_res+protein_emb_dim+1]

        # Pass through MLP
        logits = self.mlp(x)  # [B,L,1]

        return logits.squeeze(-1)  # [B,L]

# -------------------------------
# Loss
# -------------------------------
def masked_bce_loss(logits, labels, mask, pos_weight=None):
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = criterion(logits, labels)
    loss = loss * mask
    return loss.sum() / mask.sum()

# -------------------------------
# Training loop (HPC-ready)
# -------------------------------
def train_model_hpc(model, train_loader, val_loader, optimizer, pos_weight, device,
                    num_epochs=50, patience=5, save_path=None):
    best_val_loss = float('inf')
    counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss_accum = 0.0
        total_masked = 0
        for batch in train_loader:
            x = batch['embeddings'].to(device)
            y = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            optimizer.zero_grad()
            logits = model(x, mask=mask, position=batch['position'].to(device))
            loss = masked_bce_loss(logits, y, mask, pos_weight)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * mask.sum().item()
            total_masked += mask.sum().item()
        train_loss = train_loss_accum / total_masked
        train_loss_history.append(train_loss)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss_accum = 0.0
        total_masked_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['embeddings'].to(device)
                y = batch['labels'].to(device)
                mask = batch['mask'].to(device)
                logits = model(x, mask=mask, position=batch['position'].to(device))
                loss = masked_bce_loss(logits, y, mask, pos_weight)
                val_loss_accum += loss.item() * mask.sum().item()
                total_masked_val += mask.sum().item()
        val_loss = val_loss_accum / total_masked_val
        val_loss_history.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return train_loss_history, val_loss_history

# -------------------------------
# Evaluation functions
# -------------------------------
def evaluate_on_test_hpc(model, test_loader, device, threshold=0.5):
    model.eval()
    all_logits, all_labels, all_mask = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)
            logits = model(embeddings, mask, position)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    valid_logits = all_logits[all_mask.bool()]
    valid_labels = all_labels[all_mask.bool()]
    probs = torch.sigmoid(valid_logits).numpy()
    pred = (probs >= threshold).astype(int)
    true = valid_labels.numpy()
    return {
        'ROC-AUC': roc_auc_score(true, probs),
        'AU-PRC': average_precision_score(true, probs),
        'MCC': matthews_corrcoef(true, pred),
        'Precision': precision_score(true, pred),
        'Recall': recall_score(true, pred)
    }

def find_optimal_threshold(model, val_loader, device, thresholds=np.arange(0.5,1.0,0.1)):
    model.eval()
    all_logits, all_labels, all_mask = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)
            logits = model(embeddings, mask, position)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    valid_logits = all_logits[all_mask.bool()]
    valid_labels = all_labels[all_mask.bool()]
    probs = torch.sigmoid(valid_logits).numpy()
    true = valid_labels.numpy()
    best_mcc = -1.0
    best_threshold = 0.5
    results = {}
    for t in thresholds:
        pred = (probs >= t).astype(int)
        mcc = matthews_corrcoef(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        results[t] = {'MCC': mcc, 'Precision': precision, 'Recall': recall}
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t
    print(f"Optimal threshold: {best_threshold:.2f} with MCC: {best_mcc:.4f}")
    return best_threshold, results

# -------------------------------
# Execution block
# -------------------------------
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

# -------------------------------
# Saving results
# -------------------------------
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
    # Paths and data (adjust to filepath)
    # -------------------------------
    h5_path = os.path.expanduser("path/to/file/per_residue_embeddings.h5")
    train_pkl = os.path.expanduser("path/to/file/ligysis_id30_train_df.pkl")
    val_pkl = os.path.expanduser("path/to/file/ligysis_id30_val_df.pkl")
    test_pkl = os.path.expanduser("path/to/file/ligysis_id30_test_df.pkl")

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
    # Train, find optimal threshold, evaluate (Adjust path to desired location)
    # -------------------------------

    checkpoint_dir = os.path.dirname(os.path.expanduser("path/to/file/best_model.pt"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results = execute_training_pipeline_hpc(
        model, train_loader, val_loader, test_loader,
        optimizer, pos_weight, device,
        num_epochs=50,
        patience=5,
        save_path=os.path.expanduser("path/to/file/best_model.pt") # adjust path
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
