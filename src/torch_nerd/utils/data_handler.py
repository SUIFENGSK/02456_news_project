import datetime
from pathlib import Path
import polars as pl
import from_ebrec._constants as cs

import from_ebrec._behaviors as beh

def _convert_to_arr(df):
    return df.to_pandas().to_dict(orient="records")


def extract_cols(df, columns):
    cols = df.select(columns)
    return cols, _convert_to_arr(cols)


def sample_data(df, n_samples, seed=None):
    sampled_df = df.sample(n_samples, seed=seed)
    return sampled_df, _convert_to_arr(sampled_df)


class NewsDataset:
    def __init__(self):
        self.df = None
        self.df_train = None
        self.df_validation = None
        self.df_test = None
        self.df_articles = None

    def _setup_train_data(self):
        dt_split = pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
        self.df_train = self.df.filter(pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL) < dt_split)

    def _setup_validation_data(self):
        dt_split = pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
        self.df_validation = self.df.filter(pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL) >= dt_split)

    def setup_articles_data(self, dataset_path: Path):
        self.df_articles = pl.read_parquet(dataset_path.joinpath("articles.parquet"))

    def setup_test_data(self, dataset_path: Path, datasplit: str, history_size: int, columns: list[str], fraction: float, seed: int):
        self.df_test = (beh.ebnerd_from_path(dataset_path.joinpath(datasplit, "validation"), history_size=history_size, padding=0,).select(columns).pipe(beh.create_binary_labels_column).sample(fraction=fraction)
)

    def setup_df(self, dataset_path: Path, datasplit: str, history_size: int, columns: list[str], fraction: float, seed: int):
        self.df = (beh.ebnerd_from_path(dataset_path.joinpath(datasplit, "train"), history_size=history_size, padding=0,).select(columns).pipe(beh.sampling_strategy_wu2019, npratio=4, shuffle=True, with_replacement=True, seed=seed).pipe(beh.create_binary_labels_column).sample(fraction=fraction))
        self._setup_train_data()
        self._setup_validation_data()


    
