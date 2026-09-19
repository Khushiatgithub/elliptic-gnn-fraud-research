"""
Inference Service for Multi-Model Prediction and Feature Importance Analysis.
Authentically loads serialized model weights and preprocessors without fabrication.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from backend.app.config import settings
from backend.app.schemas.prediction import FeatureContribution, ModelInfo, PredictionResponse
from backend.app.services.dataset_service import DatasetService, get_dataset_service
from backend.app.services.graph_service import GraphService, get_graph_service
from src.models.gat import GATNet
from src.models.gcn import GCNNet
from src.models.graphsage import GraphSAGENet
from src.models.mlp import PyTorchMLP


class InferenceService:
    def __init__(
        self,
        dataset_service: Optional[DatasetService] = None,
        graph_service: Optional[GraphService] = None
    ):
        self.dataset_service = dataset_service or get_dataset_service()
        self.graph_service = graph_service or get_graph_service()
        
        self.tabular_dir = settings.TABULAR_CKPT_DIR
        self.gnn_dir = settings.CHECKPOINTS_DIR
        
        self.preprocessors: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self._is_loaded = False

    def load_artifacts(self):
        if self._is_loaded:
            return

        print("[InferenceService] Loading preprocessors and model checkpoints...")
        
        # 1. Preprocessors
        prep_scaled_path = self.tabular_dir / "prep_scaled_full.joblib"
        prep_unscaled_path = self.tabular_dir / "prep_unscaled_full.joblib"
        
        if prep_scaled_path.exists():
            self.preprocessors["scaled_full"] = joblib.load(prep_scaled_path)
        if prep_unscaled_path.exists():
            self.preprocessors["unscaled_full"] = joblib.load(prep_unscaled_path)

        # 2. Tabular Models
        rf_path = self.tabular_dir / "random_forest_full_seed42.joblib"
        if rf_path.exists():
            self.models["random_forest"] = joblib.load(rf_path)
            
        xgb_path = self.tabular_dir / "xgboost_full_seed42.joblib"
        if xgb_path.exists():
            self.models["xgboost"] = joblib.load(xgb_path)
            
        lr_path = self.tabular_dir / "logistic_regression_full_seed42.joblib"
        if lr_path.exists():
            self.models["logistic_regression"] = joblib.load(lr_path)

        mlp_path = self.tabular_dir / "mlp_full_seed42.pt"
        if mlp_path.exists():
            mlp_model = PyTorchMLP(in_features=165, hidden_dims=[128, 64], dropout=0.2)
            mlp_model.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True))
            mlp_model.eval()
            self.models["mlp"] = mlp_model

        # 3. GNN Models
        gat_path = self.gnn_dir / "gat_full_seed42.pt"
        if gat_path.exists():
            gat_model = GATNet(in_channels=165, hidden_channels=128, num_layers=2, heads=8, dropout=0.3)
            gat_model.load_state_dict(torch.load(gat_path, map_location="cpu", weights_only=True))
            gat_model.eval()
            self.models["gat"] = gat_model

        gcn_path = self.gnn_dir / "gcn_full_seed42.pt"
        if gcn_path.exists():
            gcn_model = GCNNet(in_channels=165, hidden_channels=128, num_layers=2, dropout=0.3)
            gcn_model.load_state_dict(torch.load(gcn_path, map_location="cpu", weights_only=True))
            gcn_model.eval()
            self.models["gcn"] = gcn_model

        graphsage_path = self.gnn_dir / "graphsage_full_seed42.pt"
        if graphsage_path.exists():
            sage_model = GraphSAGENet(in_channels=165, hidden_channels=128, num_layers=2, dropout=0.3, aggr="mean")
            sage_model.load_state_dict(torch.load(graphsage_path, map_location="cpu", weights_only=True))
            sage_model.eval()
            self.models["graphsage"] = sage_model

        self._is_loaded = True
        print(f"[InferenceService] Loaded {len(self.models)} active model engines: {list(self.models.keys())}")

    def get_available_models(self) -> List[ModelInfo]:
        self.load_artifacts()
        
        models_meta = [
            ModelInfo(
                model_id="random_forest",
                name="Random Forest",
                family="Tree Ensemble",
                is_gnn=False,
                feature_mode="Full (165)",
                optimal_threshold=0.3700,
                description="Ensemble of 100 decision trees fitted with balanced bootstrap samples.",
                available="random_forest" in self.models
            ),
            ModelInfo(
                model_id="xgboost",
                name="XGBoost",
                family="Gradient Boosting",
                is_gnn=False,
                feature_mode="Full (165)",
                optimal_threshold=0.4300,
                description="Gradient boosted decision trees optimized with histogram binning.",
                available="xgboost" in self.models
            ),
            ModelInfo(
                model_id="mlp",
                name="MLP Baseline",
                family="Neural Network",
                is_gnn=False,
                feature_mode="Full (165)",
                optimal_threshold=0.5000,
                description="Feed-forward neural network with 2 hidden layers (128-64) and BatchNorm.",
                available="mlp" in self.models
            ),
            ModelInfo(
                model_id="logistic_regression",
                name="Logistic Regression",
                family="Linear Model",
                is_gnn=False,
                feature_mode="Full (165)",
                optimal_threshold=0.5000,
                description="L2-regularized linear classification model over standardized features.",
                available="logistic_regression" in self.models
            ),
            ModelInfo(
                model_id="gat",
                name="Graph Attention Network (GAT)",
                family="Graph Neural Network",
                is_gnn=True,
                feature_mode="Full (165)",
                optimal_threshold=0.5000,
                description="2-layer GAT with 8 multi-head attention coefficients per edge.",
                available="gat" in self.models
            ),
            ModelInfo(
                model_id="graphsage",
                name="GraphSAGE",
                family="Graph Neural Network",
                is_gnn=True,
                feature_mode="Full (165)",
                optimal_threshold=0.5000,
                description="Inductive mean-aggregation message passing over local neighborhood.",
                available="graphsage" in self.models
            ),
            ModelInfo(
                model_id="gcn",
                name="Graph Convolutional Network (GCN)",
                family="Graph Neural Network",
                is_gnn=True,
                feature_mode="Full (165)",
                optimal_threshold=0.5000,
                description="Spectral-approximation graph convolution with symmetric normalization.",
                available="gcn" in self.models
            )
        ]
        return models_meta

    def predict(self, tx_id: int, model_id: str = "random_forest") -> PredictionResponse:
        self.load_artifacts()
        self.dataset_service.load_data()
        self.graph_service.load_graph()
        
        if tx_id not in self.dataset_service.tx_to_row:
            raise ValueError(f"Transaction ID {tx_id} not found in Elliptic dataset.")
            
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' is not loaded or available.")

        row_idx = self.dataset_service.tx_to_row[tx_id]
        row_data = self.dataset_service.df_features.iloc[[row_idx]]
        ts = int(row_data["time_step"].values[0])
        ground_truth = self.dataset_service.class_map.get(tx_id, "unknown")
        
        if 1 <= ts <= 34:
            split = "train"
        elif 35 <= ts <= 39:
            split = "val"
        else:
            split = "test"

        in_deg = self.graph_service.get_in_degree(tx_id)
        out_deg = self.graph_service.get_out_degree(tx_id)
        sub_1hop = self.graph_service.extract_k_hop_subgraph(tx_id, k_hop=1)
        nbr_count = len(sub_1hop.nodes) - 1

        # Determine threshold
        threshold_map = {
            "random_forest": 0.37,
            "xgboost": 0.43,
            "mlp": 0.50,
            "logistic_regression": 0.50,
            "gat": 0.50,
            "graphsage": 0.50,
            "gcn": 0.50
        }
        threshold = threshold_map.get(model_id, 0.50)

        feature_contributions: Optional[List[FeatureContribution]] = None
        graph_context: Optional[Dict[str, Any]] = None

        # Execute Prediction
        if model_id in ["random_forest", "xgboost"]:
            prep = self.preprocessors.get("unscaled_full")
            X_mat = prep.transform(row_data) if prep else row_data.iloc[:, 2:].values.astype(np.float32)
            clf = self.models[model_id]
            prob = float(clf.predict_proba(X_mat)[0])
            
            # Extract tree feature importances
            feature_names = prep.feature_names if prep else [f"feat_{i}" for i in range(165)]
            raw_importances = clf.model.feature_importances_
            
            # Sort top 8 most important features for this model
            top_indices = np.argsort(raw_importances)[::-1][:8]
            feature_contributions = []
            for rank, idx in enumerate(top_indices):
                fn = feature_names[idx]
                val = float(X_mat[0, idx])
                imp = float(raw_importances[idx])
                desc = "Local transaction statistical moment" if fn.startswith("local_") else "1-hop neighborhood aggregate feature"
                feature_contributions.append(FeatureContribution(
                    feature_name=fn,
                    value=round(val, 4),
                    importance_or_weight=round(imp, 4),
                    description=desc
                ))

        elif model_id == "logistic_regression":
            prep = self.preprocessors.get("scaled_full")
            X_mat = prep.transform(row_data) if prep else row_data.iloc[:, 2:].values.astype(np.float32)
            clf = self.models[model_id]
            prob = float(clf.predict_proba(X_mat)[0])
            
            # Linear model coefficients
            coefs = clf.model.coef_[0]
            feature_names = prep.feature_names if prep else [f"feat_{i}" for i in range(165)]
            top_indices = np.argsort(np.abs(coefs))[::-1][:8]
            feature_contributions = []
            for idx in top_indices:
                fn = feature_names[idx]
                val = float(X_mat[0, idx])
                wt = float(coefs[idx])
                desc = "Scaled linear predictor coefficient"
                feature_contributions.append(FeatureContribution(
                    feature_name=fn,
                    value=round(val, 4),
                    importance_or_weight=round(wt, 4),
                    description=desc
                ))

        elif model_id == "mlp":
            prep = self.preprocessors.get("scaled_full")
            X_mat = prep.transform(row_data) if prep else row_data.iloc[:, 2:].values.astype(np.float32)
            tensor_x = torch.tensor(X_mat, dtype=torch.float32)
            mlp_model = self.models["mlp"]
            with torch.no_grad():
                logit = mlp_model(tensor_x)
                prob = float(torch.sigmoid(logit).item())

        elif model_id in ["gat", "gcn", "graphsage"]:
            # Local subgraph inductive forward pass
            gnn_model = self.models[model_id]
            sub_2hop = self.graph_service.extract_k_hop_subgraph(tx_id, k_hop=2, max_nodes=50)
            
            # Map sub-nodes to feature rows
            node_ids = [n.tx_id for n in sub_2hop.nodes]
            center_idx = node_ids.index(tx_id)
            
            sub_rows = [self.dataset_service.tx_to_row[n_id] for n_id in node_ids]
            sub_features_df = self.dataset_service.df_features.iloc[sub_rows]
            
            prep = self.preprocessors.get("scaled_full")
            X_sub = prep.transform(sub_features_df) if prep else sub_features_df.iloc[:, 2:].values.astype(np.float32)
            
            # Edge index mapping
            id_to_sub_idx = {n_id: i for i, n_id in enumerate(node_ids)}
            edge_src = []
            edge_dst = []
            for e in sub_2hop.edges:
                if e.source_tx_id in id_to_sub_idx and e.target_tx_id in id_to_sub_idx:
                    edge_src.append(id_to_sub_idx[e.source_tx_id])
                    edge_dst.append(id_to_sub_idx[e.target_tx_id])
                    
            if len(edge_src) == 0:
                # Add self loop if isolated
                edge_src = [center_idx]
                edge_dst = [center_idx]
                
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            x_tensor = torch.tensor(X_sub, dtype=torch.float32)
            
            with torch.no_grad():
                logits = gnn_model(x_tensor, edge_index)
                prob = float(torch.sigmoid(logits[center_idx]).item())

            # Compile GNN structural context
            labeled_nbrs = [n.label for n in sub_1hop.nodes if not n.is_center and n.label != "unknown"]
            unknown_nbrs = [n.label for n in sub_1hop.nodes if not n.is_center and n.label == "unknown"]
            graph_context = {
                "subgraph_nodes_2hop": len(sub_2hop.nodes),
                "subgraph_edges_2hop": len(sub_2hop.edges),
                "direct_1hop_neighbors": nbr_count,
                "labeled_neighbors_count": len(labeled_nbrs),
                "unknown_neighbors_count": len(unknown_nbrs),
                "neighbor_class_composition": {
                    "illicit": labeled_nbrs.count("illicit"),
                    "licit": labeled_nbrs.count("licit"),
                    "unknown": len(unknown_nbrs)
                },
                "homophily_ratio": (
                    round(labeled_nbrs.count(ground_truth) / len(labeled_nbrs), 3)
                    if (len(labeled_nbrs) > 0 and ground_truth in ["licit", "illicit"])
                    else "N/A"
                )
            }

        # Format Response
        predicted_class = 1 if prob >= threshold else 0
        predicted_label = "Model prediction: Illicit" if predicted_class == 1 else "Model prediction: Licit"
        
        # Meta info
        meta_dict = {
            "random_forest": ("Random Forest", "Tree Ensemble", False),
            "xgboost": ("XGBoost", "Gradient Boosting", False),
            "mlp": ("MLP Baseline", "Neural Network", False),
            "logistic_regression": ("Logistic Regression", "Linear Model", False),
            "gat": ("Graph Attention Network (GAT)", "Graph Neural Network", True),
            "graphsage": ("GraphSAGE", "Graph Neural Network", True),
            "gcn": ("Graph Convolutional Network (GCN)", "Graph Neural Network", True)
        }
        m_name, m_fam, is_gnn = meta_dict.get(model_id, (model_id, "Unknown", False))

        return PredictionResponse(
            tx_id=tx_id,
            timestep=ts,
            split=split,
            ground_truth_label=ground_truth,
            model_id=model_id,
            model_name=m_name,
            model_family=m_fam,
            is_gnn=is_gnn,
            predicted_probability=round(prob, 4),
            classification_threshold=threshold,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence_score=round(prob if predicted_class == 1 else (1.0 - prob), 4),
            in_degree=in_deg,
            out_degree=out_deg,
            neighborhood_size_1hop=nbr_count,
            feature_contributions=feature_contributions,
            graph_context=graph_context
        )


@lru_cache()
def get_inference_service() -> InferenceService:
    service = InferenceService()
    service.load_artifacts()
    return service
