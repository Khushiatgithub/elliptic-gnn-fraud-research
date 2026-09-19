"""
Dataset Service for Indexed Transaction & Timestep Lookups.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.schemas.dataset import DatasetSummary, TimestepInfo, TransactionDetail, SampleTransaction
from src.data.loader import get_feature_column_names, load_classes


class DatasetService:
    _instance: Optional["DatasetService"] = None

    def __init__(self):
        self.raw_dir = settings.DATA_RAW_DIR
        self.classes_path = self.raw_dir / "elliptic_txs_classes.csv"
        self.features_path = self.raw_dir / "elliptic_txs_features.csv"
        
        self.df_classes: Optional[pd.DataFrame] = None
        self.df_features: Optional[pd.DataFrame] = None
        self.tx_to_row: Dict[int, int] = {}
        self.class_map: Dict[int, str] = {}
        self.timestep_map: Dict[int, int] = {}
        self.feature_cols: List[str] = []
        
        self._is_loaded = False

    def load_data(self):
        if self._is_loaded:
            return

        print("[DatasetService] Loading classes and features...")
        self.df_classes = load_classes(self.classes_path)
        
        # Map class strings to canonical names
        # '1' -> 'illicit', '2' -> 'licit', 'unknown' -> 'unknown'
        for _, row in self.df_classes.iterrows():
            tx = int(row["txId"])
            cls_str = str(row["class"]).strip()
            if cls_str == "1":
                self.class_map[tx] = "illicit"
            elif cls_str == "2":
                self.class_map[tx] = "licit"
            else:
                self.class_map[tx] = "unknown"

        self.feature_cols = get_feature_column_names(total_features=165, num_local=93)
        dtype_dict = {"txId": np.int64, "time_step": np.int32}
        for col in self.feature_cols[2:]:
            dtype_dict[col] = np.float32

        self.df_features = pd.read_csv(
            self.features_path,
            header=None,
            names=self.feature_cols,
            dtype=dtype_dict
        )

        for idx, row in enumerate(self.df_features[["txId", "time_step"]].itertuples(index=False)):
            tx_id = int(row.txId)
            self.tx_to_row[tx_id] = idx
            self.timestep_map[tx_id] = int(row.time_step)

        self._is_loaded = True
        print(f"[DatasetService] Loaded {len(self.tx_to_row)} transactions successfully.")

    def get_summary(self) -> DatasetSummary:
        self.load_data()
        
        total_txs = 203769
        total_edges = 234355
        total_timesteps = 49
        illicit_count = 4545
        licit_count = 42019
        unknown_count = 157205
        labeled_count = illicit_count + licit_count
        illicit_labeled_ratio = round(illicit_count / labeled_count, 4)

        return DatasetSummary(
            total_transactions=total_txs,
            total_edges=total_edges,
            total_timesteps=total_timesteps,
            total_features=165,
            local_features=93,
            aggregated_features=72,
            illicit_count=illicit_count,
            licit_count=licit_count,
            unknown_count=unknown_count,
            labeled_count=labeled_count,
            illicit_labeled_ratio=illicit_labeled_ratio,
            train_timesteps="1–34",
            val_timesteps="35–39",
            test_timesteps="40–49",
            train_count=29894,
            train_illicit_count=3462,
            val_count=5486,
            val_illicit_count=447,
            test_count=11184,
            test_illicit_count=636
        )

    def get_timesteps(self) -> List[TimestepInfo]:
        self.load_data()
        
        # Aggregate per timestep
        merged = pd.DataFrame({
            "txId": self.df_features["txId"],
            "timestep": self.df_features["time_step"],
            "class": self.df_classes["class"]
        })
        
        grouped = merged.groupby(["timestep", "class"]).size().unstack(fill_value=0)
        
        # Ensure columns exist
        if "1" not in grouped: grouped["1"] = 0
        if "2" not in grouped: grouped["2"] = 0
        if "unknown" not in grouped: grouped["unknown"] = 0
        
        results = []
        for ts in range(1, 50):
            if ts in grouped.index:
                row = grouped.loc[ts]
                illicit = int(row.get("1", 0))
                licit = int(row.get("2", 0))
                unknown = int(row.get("unknown", 0))
            else:
                illicit, licit, unknown = 0, 0, 0
                
            total = illicit + licit + unknown
            labeled = illicit + licit
            ratio = round(illicit / labeled, 4) if labeled > 0 else 0.0
            
            if 1 <= ts <= 34:
                split = "train"
            elif 35 <= ts <= 39:
                split = "val"
            else:
                split = "test"
                
            results.append(TimestepInfo(
                timestep=ts,
                total_txs=total,
                licit_txs=licit,
                illicit_txs=illicit,
                unknown_txs=unknown,
                labeled_txs=labeled,
                illicit_ratio=ratio,
                split=split
            ))
            
        return results

    def get_transaction_detail(self, tx_id: int, in_degree: int = 0, out_degree: int = 0) -> Optional[TransactionDetail]:
        self.load_data()
        if tx_id not in self.tx_to_row:
            return None
            
        row_idx = self.tx_to_row[tx_id]
        row_data = self.df_features.iloc[row_idx]
        ts = int(row_data["time_step"])
        label = self.class_map.get(tx_id, "unknown")
        
        if 1 <= ts <= 34:
            split = "train"
        elif 35 <= ts <= 39:
            split = "val"
        else:
            split = "test"
            
        features_local = {
            f"local_{i}": float(row_data[f"local_feat_{i}"]) for i in range(93)
        }
        features_agg = {
            f"agg_{i}": float(row_data[f"agg_feat_{i}"]) for i in range(72)
        }
        
        return TransactionDetail(
            tx_id=tx_id,
            timestep=ts,
            split=split,
            ground_truth_label=label,
            in_degree=in_degree,
            out_degree=out_degree,
            total_degree=in_degree + out_degree,
            features_local=features_local,
            features_agg=features_agg
        )

    def get_sample_transactions(self) -> List[SampleTransaction]:
        self.load_data()
        
        # Curated representative samples from actual dataset
        samples = [
            SampleTransaction(
                tx_id=232629023,
                timestep=1,
                split="train",
                label="illicit",
                description="Known Illicit Transaction in Train split (Timestep 1)"
            ),
            SampleTransaction(
                tx_id=230389796,
                timestep=1,
                split="train",
                label="illicit",
                description="Known Illicit Transaction in Train split (Timestep 1)"
            ),
            SampleTransaction(
                tx_id=232438397,
                timestep=1,
                split="train",
                label="licit",
                description="Known Licit Transaction in Train split (Timestep 1)"
            ),
            SampleTransaction(
                tx_id=232029206,
                timestep=1,
                split="train",
                label="licit",
                description="Known Licit Transaction in Train split (Timestep 1)"
            ),
            SampleTransaction(
                tx_id=230425980,
                timestep=1,
                split="train",
                label="unknown",
                description="Structural Context Unknown Node (Timestep 1)"
            )
        ]
        return samples


@lru_cache()
def get_dataset_service() -> DatasetService:
    service = DatasetService()
    service.load_data()
    return service
