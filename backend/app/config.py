"""
Application Configuration and Path Settings.
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "Elliptic GNN vs Tabular Research Demonstration API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Root directory (points to elliptic-gnn-fraud-research/)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    @property
    def DATA_RAW_DIR(self) -> Path:
        return self.BASE_DIR / "data" / "raw" / "elliptic_bitcoin_dataset"
        
    @property
    def TABLES_DIR(self) -> Path:
        return self.BASE_DIR / "results" / "tables"
        
    @property
    def FIGURES_DIR(self) -> Path:
        return self.BASE_DIR / "results" / "figures"
        
    @property
    def CHECKPOINTS_DIR(self) -> Path:
        return self.BASE_DIR / "models" / "checkpoints"
        
    @property
    def TABULAR_CKPT_DIR(self) -> Path:
        return self.BASE_DIR / "models" / "checkpoints" / "tabular"
        
    @property
    def CONFIGS_DIR(self) -> Path:
        return self.BASE_DIR / "configs"
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "*"
    ]
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
