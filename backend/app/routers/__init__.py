from .dataset import router as dataset_router
from .graph import router as graph_router
from .results import router as results_router
from .prediction import router as prediction_router
from .figures import router as figures_router

__all__ = [
    "dataset_router",
    "graph_router",
    "results_router",
    "prediction_router",
    "figures_router"
]
