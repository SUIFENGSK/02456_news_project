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
    def __init__(self, dataset_path: Path):
        self.df = None
        self.dataset_path = dataset_path
        self.df_behaviors_train = self._load_behaviors_train()
        self.df_history_train = self._load_history_train()
        self.df_behaviors_validation = self._load_behaviors_validation()
        self.df_history_validation = self._load_history_validation()
        self.df_articles = self._load_articles()

    def _setup_train_data(self):
        dt_split = pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
        self.df_train = self.df.filter(pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL) < dt_split)

    def _setup_validation_data(self):
        dt_split = pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
        self.df_validation = self.df.filter(pl.col(cs.DEFAULT_IMPRESSION_TIMESTAMP_COL) >= dt_split)

    def setup_df(self, dataset_path: Path, datasplit: str, history_size: int, columns: list[str], fraction: float, seed: int):
        self.df = (beh.ebnerd_from_path(dataset_path.joinpath(datasplit, "train"), history_size=history_size, padding=0,).select(columns).pipe(beh.sampling_strategy_wu2019, npratio=4, shuffle=True, with_replacement=True, seed=seed).pipe(beh.create_binary_labels_column).sample(fraction=fraction))
        self._setup_train_data()
        self._setup_validation_data()

    def _load_behaviors_train(self):
        return pl.read_parquet(self.dataset_path.joinpath("train/behaviors.parquet"))

    def _load_history_train(self):
        return pl.read_parquet(self.dataset_path.joinpath("train/history.parquet"))

    def _load_behaviors_validation(self):
        return pl.read_parquet(self.dataset_path.joinpath("validation/behaviors.parquet"))

    def _load_history_validation(self):
        return pl.read_parquet(self.dataset_path.joinpath("validation/history.parquet"))

    def _load_articles(self):
        return pl.read_parquet(self.dataset_path.joinpath("articles.parquet"))

    def test_data_loading(self):
        # Print basic info about the loaded dataframes
        print("Train Behaviors DataFrame:")
        print(self.df_behaviors_train.shape)
        print(self.df_behaviors_train.head())

        print("\nTrain History DataFrame:")
        print(self.df_history_train.shape)
        print(self.df_history_train.head())

        print("\nValidation Behaviors DataFrame:")
        print(self.df_behaviors_validation.shape)
        print(self.df_behaviors_validation.head())

        print("\nValidation History DataFrame:")
        print(self.df_history_validation.shape)
        print(self.df_history_validation.head())

        print("\nArticles DataFrame:")
        print(self.df_articles.shape)
        print(self.df_articles.head())

    def get_article_info_by_id(self, article_id):
        # Filter the articles DataFrame for the given article ID
        article_info = self.df_articles.filter(pl.col(cs.DEFAULT_ARTICLE_ID_COL) == int(article_id))

        # Convert the result to a dictionary format (or handle it as needed)
        return article_info, _convert_to_arr(article_info)