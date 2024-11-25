from pathlib import Path
import polars as pl

class NewsDataset:
    def __init__(self, dataset_path: Path):

        self.dataset_path = dataset_path
        self.df_behaviors_train = self._load_behaviors_train()
        self.df_history_train = self._load_history_train()
        self.df_behaviors_validation = self._load_behaviors_validation()
        self.df_history_validation = self._load_history_validation()
        self.df_articles = self._load_articles()



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