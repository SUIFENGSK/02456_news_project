from dataclasses import dataclass, field
from datetime import datetime
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
    DEFAULT_IMPRESSION_TIMESTAMP_COL
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

    def __getitem__(self, idx):
        # Get the current batch of data
        batch_x = self.X[idx * self.batch_size: (idx + 1) * self.batch_size].pipe(self.transform)
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Convert labels to tensors
        batch_y = torch.tensor(batch_y.to_list(), dtype=torch.float32)

        # Prepare input features
        his_input_title = self.lookup_article_matrix[batch_x[self.history_column].to_list()]
        pred_input_title = self.lookup_article_matrix[batch_x[self.inview_col].to_list()]
        pred_input_title = np.squeeze(pred_input_title, axis=2)
        his_input_title = np.squeeze(his_input_title, axis=2)

        # Extract the impression timestamps
        if DEFAULT_IMPRESSION_TIMESTAMP_COL in batch_x.columns:
            timestamps = batch_x[DEFAULT_IMPRESSION_TIMESTAMP_COL].apply(
            lambda x: x.timestamp() if isinstance(x, datetime) else float("nan")
        ).to_list()
            timestamps = torch.tensor(timestamps, dtype=torch.float32)  # Convert to tensor
        else:
            timestamps = None  # Handle the case where timestamps might not exist

        # Return inputs and labels, including timestamps
        return (torch.tensor(his_input_title, dtype=torch.float32),
                torch.tensor(pred_input_title, dtype=torch.float32),
                timestamps), batch_y

