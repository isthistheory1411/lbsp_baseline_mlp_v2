import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

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
