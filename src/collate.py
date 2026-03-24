import torch

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
