import pandas as pd
from pathlib import Path

class Reader:
    def __init__(self, dataset_path: Path, data_set: str):
        self.path = dataset_path.joinpath(data_set)
        
    def read(self, purpose: str):
        if purpose not in ["train", "validation"]:
            raise ValueError("Purpose must be either 'train', 'validation'")
        behaviours = pd.read_parquet(self.path.joinpath(purpose, "behaviors.parquet"))
        history = pd.read_parquet(self.path.joinpath(purpose, "history.parquet"))
        articles = pd.read_parquet(self.path.joinpath("articles.parquet"))
        return behaviours, history, articles
        

        
