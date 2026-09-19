"""
Multilayer Perceptron (MLP) Neural Network Baseline in PyTorch.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger("mlp_model")


class PyTorchMLP(nn.Module):
    """Deep Feed-Forward Neural Network Architecture."""

    def __init__(self, in_features: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = in_features
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPBaseline:
    """
    MLP Baseline classifier supporting standard BCE and Pos-Weight Cost-Sensitive Loss.
    """

    def __init__(
        self,
        weighted: bool = False,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 40,
        random_state: int = 42
    ):
        self.weighted = weighted
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PyTorchMLP] = None
        self.pos_weight: float = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MLPBaseline":
        """Train MLP model on training data."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        in_features = X_train.shape[1]
        self.model = PyTorchMLP(
            in_features=in_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss function with optional positive class weighting
        if self.weighted:
            n_licit = float((y_train == 0).sum())
            n_illicit = float((y_train == 1).sum())
            self.pos_weight = float(n_licit / n_illicit) if n_illicit > 0 else 1.0
            pos_weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.pos_weight = 1.0
            criterion = nn.BCEWithLogitsLoss()
            
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Data loader
        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute positive class probabilities using sigmoid."""
        if self.model is None:
            raise RuntimeError("MLP model must be fitted before predict_proba.")
            
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using specified threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "model_type": "PyTorch_MLP",
            "weighted": self.weighted,
            "pos_weight": float(self.pos_weight),
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "random_state": self.random_state
        }
