"""
Figure Metadata Pydantic Schemas.
"""

from typing import Optional
from pydantic import BaseModel


class FigureInfo(BaseModel):
    filename: str
    title: str
    category: str
    description: str
    paper_section: str
    url: str
