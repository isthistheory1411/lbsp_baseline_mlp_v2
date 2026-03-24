import torch
import torch.nn as nn

def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor = None
) -> torch.Tensor:
    """
    Binary Cross-Entropy loss per residue, ignoring masked residues.

    Args:
        logits: [B, L] model outputs before sigmoid
        labels: [B, L] ground truth labels (0/1)
        mask: [B, L] 1 for valid residues, 0 for padded residues
        pos_weight: optional weighting for positive class

    Returns:
        Scalar tensor: mean BCE over valid residues
    """
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = criterion(logits, labels)
    loss = loss * mask  # zero out padded residues
    return loss.sum() / mask.sum()  # mean over valid residues
