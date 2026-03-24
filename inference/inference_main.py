import os
import torch
import joblib
import argparse
import json
from omegaconf import OmegaConf
from src.model import ResidueMLP
from src.utils import set_seed  
from inference.inference import run_inference  

if __name__ == "__main__":
    # -------------------------------
    # Command-line arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="Run per-residue inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference_config.yaml")
    parser.add_argument("--override", nargs="*", default=None,
                        help="Optional overrides, e.g. inference.threshold=0.6")
    args = parser.parse_args()

    # -------------------------------
    # Load config
    # -------------------------------
    cfg = OmegaConf.load(os.path.expanduser(args.config))
    if args.override:
        for override in args.override:
            key, value = override.split("=")
            OmegaConf.update(cfg, key, value)

    # -------------------------------
    # Set random seed
    # -------------------------------
    if hasattr(cfg.inference, "seed"):
        set_seed(cfg.inference.seed)

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Load model
    # -------------------------------
    model = ResidueMLP(
        residue_emb_dim=cfg.model.residue_emb_dim,
        protein_emb_dim=cfg.model.protein_emb_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout
    )
    checkpoint_path = os.path.expanduser(cfg.paths.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    # -------------------------------
    # Load dataset
    # -------------------------------
    df = joblib.load(os.path.expanduser(cfg.data.test_df))
    h5_path = os.path.expanduser(cfg.data.h5_embeddings)

    # -------------------------------
    # Output CSV path
    # -------------------------------
    save_csv = os.path.expanduser(cfg.paths.inference_csv)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)

    # -------------------------------
    # Run inference
    # -------------------------------
    results = run_inference(
        model=model,
        df=df,
        h5_path=h5_path,
        device=device,
        batch_size=cfg.inference.batch_size,
        max_len=cfg.model.max_len,
        threshold=cfg.inference.threshold,
        save_csv=save_csv
    )

    print(f"Inference complete. Predictions saved to {save_csv}")

    # -------------------------------
    # Save metrics if available
    # -------------------------------
    if "metrics" in results:
        metrics_path = os.path.splitext(save_csv)[0] + "_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=4)
        print(f"Inference metrics computed and saved to {metrics_path}")
