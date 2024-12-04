import torch
from torch.utils.data import Dataset
import numpy as np

from from_ebrec import _constants as cs

import pandas as pd


class NRMSDataset(Dataset):
    def __init__(self, 
                 behaviours : pd.DataFrame,
                 history : pd.DataFrame,
                 articles : pd.DataFrame,
                 fraction: float,
                 history_size: int = 50,
                 ): 
        
        # Load the data

        # Data should be a list of dictionaries, where each dictionary contains the following
        # keys:
        # - "history": a numpy array of shape (history_size, title_size, embedding_dim)
        # - "candidates": a numpy array of shape (candidate_size, title_size, embedding_dim)

        # select the columns in the behaviours dataframe
        behaviours = behaviours[[cs.DEFAULT_USER_COL, cs.DEFAULT_INVIEW_ARTICLES_COL, cs.DEFAULT_CLICKED_ARTICLES_COL, cs.DEFAULT_IMPRESSION_TIMESTAMP_COL]]

        # sample a fraction of the behaviours dataframe
        behaviours = behaviours.sample(frac = fraction)

        articles = articles[[cs.DEFAULT_ARTICLE_ID_COL, cs.DEFAULT_TITLE_COL, cs.DEFAULT_SUBTITLE_COL]]

        # Merge the dataframes based on the article ids, i.e. expand the inview articles and clicked articles to include the title and subtitle from the articles dataframe
        behaviours = behaviours.explode(cs.DEFAULT_INVIEW_ARTICLES_COL)
        behaviours = behaviours.explode(cs.DEFAULT_CLICKED_ARTICLES_COL)

        behaviours = behaviours.merge(articles, left_on = cs.DEFAULT_INVIEW_ARTICLES_COL, right_on = cs.DEFAULT_ARTICLE_ID_COL, how = "left")
        behaviours = behaviours.merge(articles, left_on = cs.DEFAULT_CLICKED_ARTICLES_COL, right_on = cs.DEFAULT_ARTICLE_ID_COL, how = "left")

        # Drop the article id columns
        behaviours = behaviours.drop(columns = [cs.DEFAULT_ARTICLE_ID_COL + "_x", cs.DEFAULT_ARTICLE_ID_COL + "_y"])

        # Rename the columns
        behaviours = behaviours.rename(columns = {cs.DEFAULT_TITLE_COL + "_x": cs.DEFAULT_TITLE_COL + "_inview", cs.DEFAULT_SUBTITLE_COL + "_x": cs.DEFAULT_SUBTITLE_COL + "_inview", cs.DEFAULT_TITLE_COL + "_y": cs.DEFAULT_TITLE_COL + "_clicked", cs.DEFAULT_SUBTITLE_COL + "_y": cs.DEFAULT_SUBTITLE_COL + "_clicked"})

        # Fill the NaN values with empty strings
        behaviours = behaviours.fillna("")

        # Group the behaviours by the user id and the impression timestamp
        behaviours = behaviours.groupby([cs.DEFAULT_USER_COL, cs.DEFAULT_IMPRESSION_TIMESTAMP_COL]).agg({cs.DEFAULT_INVIEW_ARTICLES_COL: list, cs.DEFAULT_CLICKED_ARTICLES_COL: list, cs.DEFAULT_TITLE_COL + "_inview": list, cs.DEFAULT_SUBTITLE_COL + "_inview": list, cs.DEFAULT_TITLE_COL + "_clicked": list, cs.DEFAULT_SUBTITLE_COL + "_clicked": list}).reset_index()

        # Sort the behaviours by the user id and the impression timestamp
        behaviours = behaviours.sort_values([cs.DEFAULT_USER_COL, cs.DEFAULT_IMPRESSION_TIMESTAMP_COL])

        # ensure title_clicked and subtitle clicked only contain one element
        behaviours[cs.DEFAULT_TITLE_COL + "_clicked"] = behaviours[cs.DEFAULT_TITLE_COL + "_clicked"].apply(lambda x: x[0] if len(x) > 0 else "")
        behaviours[cs.DEFAULT_SUBTITLE_COL + "_clicked"] = behaviours[cs.DEFAULT_SUBTITLE_COL + "_clicked"].apply(lambda x: x[0] if len(x) > 0 else "")
      
        pd.set_option('display.max_columns', 10)
        # print only the first row title_inview column
        #print(behaviours[cs.DEFAULT_TITLE_COL + "_inview"].iloc[0])
        # print only the first row title_clicked column
        #print(behaviours[cs.DEFAULT_TITLE_COL + "_clicked"].iloc[0])

        #print(behaviours.head())
        print("Handling history")
        history = history[["user_id", cs.DEFAULT_HISTORY_ARTICLE_ID_COL, cs.DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL]]
        
        # get the type of the history article id column
        print(history[cs.DEFAULT_HISTORY_ARTICLE_ID_COL].dtype)

        print(history.head())
        



        self.data = []
        self.labels = []
        



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # should return a tuple of inputs and labels for a single batch
        data = self.data[idx]
        labels = self.labels[idx]

        # Prepare input features
        history = data["history"]
        candidates = data["candidates"]

        # Convert to PyTorch tensors
        history = torch.tensor(history, dtype=torch.float32)
        candidates = torch.tensor(candidates, dtype=torch.float32)
        labels = torch.tensor(labels.to_list(), dtype=torch.float32)

        # Return inputs and labels
        inputs = (history, candidates)
        return (inputs, labels)


from parquets import Reader
from pathlib import Path

DATAPATH = Path("~/ebnerd_data").expanduser()
DATASET = "ebnerd_demo"

parquet_reader = Reader(dataset_path = DATAPATH, data_set = DATASET)
behaviours, history, articles = parquet_reader.read("train")

nrms_data = NRMSDataset(behaviours, history, articles, 0.001)
