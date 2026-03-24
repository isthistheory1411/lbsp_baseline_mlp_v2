import torch
import torch.nn as nn


class ResidueMLP(nn.Module):
    """
    Per-residue MLP for ligand binding site prediction.

    Combines:
        - per-residue embeddings
        - projected per-protein embedding
        - normalized residue position

    Supports configurable hidden layers, dropout, and LayerNorm.
    """
    def __init__(
        self,
        residue_emb_dim: int,
        protein_emb_dim: int = 256,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        self.protein_proj = nn.Linear(residue_emb_dim, protein_emb_dim)

        # MLP layers
        layers = []
        input_dim = residue_emb_dim + protein_emb_dim + 1  # per-residue + protein + position
        last_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            last_dim = h

        layers.append(nn.Linear(last_dim, 1))  # final output logit per residue
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,  # [B,L,D_res]
        mask: torch.Tensor,        # [B,L]
        position: torch.Tensor     # [B,L,1]
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            logits: [B,L] per-residue binding scores
        """
        # Compute per-protein mean embedding over valid residues
        mask_exp = mask.unsqueeze(-1)                 # [B,L,1]
        sum_emb = (embeddings * mask_exp).sum(dim=1) # [B,D_res]
        lengths = mask.sum(dim=1, keepdim=True)      # [B,1]
        mean_emb = sum_emb / lengths                 # [B,D_res]

        # Project to per-protein embedding
        protein_feats = self.protein_proj(mean_emb)  # [B, protein_emb_dim]

        # Expand per-protein embedding to sequence length
        protein_feats_exp = protein_feats.unsqueeze(1).expand(-1, embeddings.size(1), -1)  # [B,L,protein_emb_dim]

        # Concatenate per-residue embedding + projected protein embedding + position
        x = torch.cat([embeddings, protein_feats_exp, position], dim=-1)  # [B,L,D_res+protein_emb_dim+1]

        # Pass through MLP
        logits = self.mlp(x)  # [B,L,1]

        return logits.squeeze(-1)  # [B,L]
