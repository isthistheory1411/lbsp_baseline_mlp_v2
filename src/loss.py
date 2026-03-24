import torch
import torch.nn as nn

def masked_bce_loss(logits, labels, mask, pos_weight=None):
    """
    BCE loss per residue, ignoring masked residues.
    """
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = criterion(logits, labels)
    loss = loss * mask
    return loss.sum() / mask.sum()
