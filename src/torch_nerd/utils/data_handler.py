from pathlib import Path
import polars as pl
import utils.constants as cs

def _convert_to_arr(df):
    return df.to_pandas().to_dict(orient="records")


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

    def extract_history_train_data(self, columns):
        return _convert_to_arr(self.df_history_train.select(columns))

    def extract_behaviors_train_data(self, columns):
        return _convert_to_arr(self.df_behaviors_train.select(columns))

    def extract_history_validation_data(self, columns):
        return _convert_to_arr(self.df_history_validation.select(columns))

    def extract_behaviors_validation_data(self, columns):
        return _convert_to_arr(self.df_behaviors_validation.select(columns))

    def extract_articles_data(self, columns):
        return _convert_to_arr(self.df_articles.select(columns))

    def get_article_info_by_id(self, article_id):
        # Filter the articles DataFrame for the given article ID
        article_info = self.df_articles.filter(pl.col(cs.DEFAULT_ARTICLE_ID_COL) == int(article_id))

        # Convert the result to a dictionary format (or handle it as needed)
        return _convert_to_arr(article_info)

