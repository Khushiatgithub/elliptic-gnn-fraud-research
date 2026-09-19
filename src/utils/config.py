"""
Configuration loader and constants for Elliptic GNN Fraud Research.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """Return project root directory based on file location."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml relative to project root.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        
    if not config_path.exists():
        # Return fallback configuration dictionary
        return {
            "project": {"name": "elliptic-gnn-fraud-research", "random_seed": 42},
            "paths": {
                "raw_dir": str(root / "data" / "raw" / "elliptic_bitcoin_dataset"),
                "processed_dir": str(root / "data" / "processed"),
                "results_tables_dir": str(root / "results" / "tables"),
                "results_figures_dir": str(root / "results" / "figures"),
                "results_logs_dir": str(root / "results" / "logs"),
                "classes_file": str(root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv"),
                "edgelist_file": str(root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_edgelist.csv"),
                "features_file": str(root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv"),
            },
            "dataset": {
                "num_timesteps": 49,
                "total_features": 165,
                "num_local_features": 93,
                "num_agg_features": 72,
            },
            "visualization": {"dpi": 300},
        }
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    # Resolve relative paths relative to project root
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not Path(val).is_absolute():
            cfg["paths"][key] = str(root / val)
            
    return cfg
