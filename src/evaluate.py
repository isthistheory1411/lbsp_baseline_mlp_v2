import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score
import numpy as np


def evaluate_on_test_hpc(model, test_loader, device, threshold=0.5):
      """
    Evaluate trained ResidueMLP on test set (HPC-ready).

    Args:
        model: trained ResidueMLP
        test_loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        threshold: probability threshold for binary classification

    Returns:
        metrics: dict containing ROC-AUC, AU-PRC, MCC, precision, recall
    """
    model.eval()
    all_logits, all_labels, all_mask = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)
          
            logits = model(embeddings, mask, position)  # [B,L]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())

    # Concatenate batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    
    # Only valid residues
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
     """
    Sweep thresholds on validation set to find the threshold that maximizes MCC.
    Default sweep: 0.5, 0.6, 0.7, 0.8, 0.9.

    Args:
        model: trained ResidueMLP
        val_loader: validation DataLoader
        device: 'cuda' or 'cpu'
        thresholds: array of probability thresholds to test

    Returns:
        best_threshold: threshold maximizing MCC
        results: dict mapping threshold -> metrics
    """
    model.eval()
    all_logits, all_labels, all_mask = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass using updated ResidueMLP
            logits = model(embeddings, mask, position)

            # Move to CPU to save GPU memory
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())
        
    # Concatenate all batches  
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask = torch.cat(all_mask, dim=0)

    # Flatten only valid residues
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
