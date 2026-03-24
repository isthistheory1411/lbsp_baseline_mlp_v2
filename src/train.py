import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

def train_model_hpc(model, train_loader, val_loader, optimizer,
                    pos_weight, device, num_epochs=50,
                    patience=5, save_path="best_model.pt",
                    use_amp=False):
    """
    HPC-ready training loop for ResidueMLP.
    
    Args:
        model: ResidueMLP instance
        train_loader, val_loader: PyTorch DataLoaders
        optimizer: torch optimizer
        pos_weight: scalar tensor for BCEWithLogitsLoss
        device: 'cuda' or 'cpu'
        num_epochs: maximum epochs to train
        patience: early stopping patience
        save_path: where to save the best model
        use_amp: use mixed precision (torch.cuda.amp) if True
    Returns:
        train_loss_history, val_loss_history
    """
    best_val_loss = float('inf')
    counter = 0

    train_loss_history = []
    val_loss_history = []

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, num_epochs + 1):
        # --------------------
        # Training step
        # --------------------
        model.train()
        train_loss_accum = 0.0
        total_masked = 0

        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(embeddings, mask, position)
                    loss = masked_bce_loss(logits, labels, mask, pos_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(embeddings, mask, position)
                loss = masked_bce_loss(logits, labels, mask, pos_weight)
                loss.backward()
                optimizer.step()

            train_loss_accum += loss.item() * mask.sum().item()
            total_masked += mask.sum().item()

        train_loss = train_loss_accum / total_masked
        train_loss_history.append(train_loss)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # --------------------
        # Validation step
        # --------------------
        model.eval()
        val_loss_accum = 0.0
        total_masked_val = 0

        all_logits, all_labels, all_mask = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                mask = batch['mask'].to(device)
                position = batch['position'].to(device)
                labels = batch['labels'].to(device)

                logits = model(embeddings, mask, position)
                loss = masked_bce_loss(logits, labels, mask, pos_weight)

                val_loss_accum += loss.item() * mask.sum().item()
                total_masked_val += mask.sum().item()

                # Collect for metrics
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_mask.append(mask.cpu())

        val_loss = val_loss_accum / total_masked_val
        val_loss_history.append(val_loss)

        # --------------------
        # Flatten valid residues for metrics
        # --------------------
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_mask = torch.cat(all_mask, dim=0)

        valid_logits = all_logits[all_mask.bool()]
        valid_labels = all_labels[all_mask.bool()]

        probs = torch.sigmoid(valid_logits).numpy()
        true = valid_labels.numpy()
        pred = (probs >= 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(true, probs)
        auprc = average_precision_score(true, probs)
        mcc = matthews_corrcoef(true, pred)

        print(f"Validation | Loss: {val_loss:.4f} | ROC-AUC: {auc:.4f} | "
              f"AU-PRC: {auprc:.4f} | MCC: {mcc:.4f}")

        # --------------------
        # Early stopping
        # --------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            counter += 1
            print(f"No improvement. Patience counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return train_loss_history, val_loss_history
