# --- START OF MODIFIED FILE nn.py ---

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.5,
        activation: str = "leakyrelu"
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            if activation.lower() == "leakyrelu":
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TorchWrapper:
    """
    Wraps a trained PyTorch model to expose prediction methods.
    Includes method to get raw logits.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval() 

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Predicts raw logits (scores) before thresholding."""
        self.model.eval() 
        with torch.no_grad():
            x_tensor = torch.from_numpy(X.astype(np.float32))
            
            logits = self.model(x_tensor).squeeze()
            if logits.dim() > 1:
                 # Assuming batch dimension is first, squeeze other dims of size 1
                 logits = logits.squeeze(dim=-1)
            elif logits.dim() == 0: # Handle single prediction case
                 logits = logits.unsqueeze(0)

        return logits.cpu().numpy() 

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Predicts class labels {-1, 1} using a specified threshold."""
        logits = self.predict_logits(X)
        return np.where(logits >= threshold, 1, -1)

    def predict_submission(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Predicts class labels {0, 1} using a specified threshold."""
       
        preds_m1p1 = self.predict(X, threshold=threshold)
        return np.where(preds_m1p1 == -1, 0, 1)


def build_model_nn(X, y, params):
    """
    Trains an MLP on (X, y) and returns a TorchWrapper.
    params keys:
      - hidden_dims (list[int]) # Changed tuple to list for consistency
      - lr          (float)
      - epochs      (int)
      - batch_size  (int)
      - dropout     (float)
      - optimizer   (str, 'adam' or 'adamw')
      - weight_decay (float)
      - random_state (optional, int)
      - early_stopping_patience (optional, int, default 10)
      - activation (optional, str, default "leakyrelu")
    Expects y in {-1, 1} format for splitting, converts internally to {0, 1} for loss.
    """
    # reproducibility
    seed = params.get("random_state")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Add for GPU reproducibility if applicable
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    input_dim = X.shape[1]
    hidden_dims = params.get("hidden_dims", [128, 64]) # Use list consistently
    lr = params.get("lr", 1e-3)
    epochs = params.get("epochs", 50)
    batch_size = params.get("batch_size", 32)
    dropout = params.get("dropout", 0.5)
    opt_name = params.get("optimizer", "adam").lower()
    weight_decay = params.get("weight_decay", 0.0)
    activation = params.get("activation", "leakyrelu")
    max_no_improve = params.get("early_stopping_patience", 10)

    # train/validation split for early stopping
    # Stratify requires original class labels {0, 1} or consistent {-1, 1}
    # Assuming y is passed in {-1, 1}, stratify should still work correctly
    # Use a fixed random state for this internal split for consistency if seed is provided
    val_split_seed = seed + 1 if seed is not None else None # Use different seed for split
    X_train, X_val, y_train_m1p1, y_val_m1p1 = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=val_split_seed
    )

    # mapping y from {-1,1} to {0,1} for BCEWithLogitsLoss
    y_train01 = ((y_train_m1p1 + 1) / 2).astype(np.float32)
    y_val01 = ((y_val_m1p1 + 1) / 2).astype(np.float32)

    model = MLP(input_dim, list(hidden_dims), dropout=dropout, activation=activation) # Ensure list
    criterion = nn.BCEWithLogitsLoss()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) # verbose=False to reduce noise


    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train01.reshape(-1,1))
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val01.reshape(-1,1))
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False) 

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    #print(f"Starting training for max {epochs} epochs...") 
    for epoch in range(epochs):
        model.train() #e
        total_train_loss = 0.0
        for xb, yb in train_loader:
            # xb, yb = xb.to(device), yb.to(device) # Move data to device
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # validation
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                # xb, yb = xb.to(device), yb.to(device) # Move data to device
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr'] # Get current LR
        scheduler.step(avg_val_loss)


        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     print(f"Epoch {epoch+1}/{epochs}.. Train Loss: {avg_train_loss:.4f}.. Val Loss: {avg_val_loss:.4f}.. LR: {current_lr:.1e}")

        # early stopping
        if avg_val_loss < best_loss - 1e-4: # small delta to prevent trivial saves
            best_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= max_no_improve:
                #print(f"Early stopping triggered at epoch {epoch + 1} after {max_no_improve} epochs with no improvement.")
                break

    # restore best model state
    if best_state is not None:
        #print(f"Restoring best model state with validation loss: {best_loss:.4f}")
        model.load_state_dict(best_state)
    else:
        print("Warning: No best model state found (early stopping patience might be too low or training too short). Using final model state.")

    return TorchWrapper(model)

