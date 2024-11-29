from dataclasses import dataclass, field
from torch.utils.data import Dataset
import torch
import numpy as np
import polars as pl

from from_ebrec._articles_behaviors import map_list_article_id_to_value
from from_ebrec._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)
from from_ebrec._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)


@dataclass
class NewsrecDataLoader(Dataset):
    """
    A PyTorch Dataset for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        raise NotImplementedError("Function '__getitem__' needs to be implemented.")

    def load_data(self):
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class NRMSDataLoader(NewsrecDataLoader):
    def transform(self, df):
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx): # Returns a tuple of input and output tensors. Input is a tuple of previous articles and current articles, output is the click label.
        batch_X = self.X[idx * self.batch_size: (idx + 1) * self.batch_size].pipe(self.transform) # Get the batch of data
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size] # Get the batch of labels
        
        if self.eval_mode: # TODO DOES NOT WORK
            repeats = torch.tensor(batch_X["n_samples"].to_list(), dtype=torch.long) 
            batch_y = torch.tensor(batch_y.explode().to_list(), dtype=torch.float32).reshape(-1, 1) 
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats.numpy(),
            )
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = torch.tensor(batch_y.to_list(), dtype=torch.float32) # Convert the labels to a tensor of floats
            his_input_title = self.lookup_article_matrix[    # Get the previous articles 
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[   # Get the current articles
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2) 

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (torch.tensor(his_input_title, dtype=torch.float32), 
                torch.tensor(pred_input_title, dtype=torch.float32)), batch_y
