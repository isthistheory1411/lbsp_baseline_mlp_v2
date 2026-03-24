import torch
import torch.nn as nn

class ResidueMLP(nn.Module):
    def __init__(self, residue_emb_dim, protein_emb_dim=256, hidden_dims=[512, 256, 128], dropout=0.1):
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

        layers = []
        input_dim = residue_emb_dim + protein_emb_dim + 1  # per-residue + protein + position
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LayerNorm(h))       # <-- re-added LayerNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))   # keep dropout
            last_dim = h

        layers.append(nn.Linear(last_dim, 1))      # output logit per residue
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings, mask, position):
        """
        embeddings: [B,L,D_res] per-residue embeddings
        mask: [B,L] mask of valid residues
        position: [B,L,1] normalized residue positions
        
        returns: [B,L] logits per residue
        """
        # Compute per-protein mean embedding over valid residues
        mask_exp = mask.unsqueeze(-1)                     # [B,L,1]
        sum_emb = (embeddings * mask_exp).sum(dim=1)     # [B,D_res]
        lengths = mask.sum(dim=1, keepdim=True)          # [B,1]
        mean_emb = sum_emb / lengths                     # [B,D_res]

        # Linear projection to smaller per-protein embedding
        protein_feats = self.protein_proj(mean_emb)      # [B, protein_emb_dim]

        # Expand per-protein embedding to sequence length
        protein_feats_exp = protein_feats.unsqueeze(1).expand(-1, embeddings.size(1), -1)  # [B,L,protein_emb_dim]

        # Concatenate per-residue embedding + projected protein embedding + position
        x = torch.cat([embeddings, protein_feats_exp, position], dim=-1)  # [B,L,D_res+protein_emb_dim+1]

        # Pass through MLP
        logits = self.mlp(x)  # [B,L,1]

        return logits.squeeze(-1)  # [B,L]
