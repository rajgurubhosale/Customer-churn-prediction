import pandas as pd
import numpy as np
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.logger import logger
from src.exception import MyException
from src.utils.main_utils import read_config_file, save_object


class FeatureEngineer:
    """Handles feature creation and removal"""

    def __init__(self, cfg):
        self.cfg = cfg

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['total_minutes'] = (
                df['Total day minutes'] +
                df['Total eve minutes'] +
                df['Total night minutes']
            )
            df['total_charge'] = (
                df['Total day charge'] +
                df['Total eve charge'] +
                df['Total night charge']
            )
            df['avg_calls'] = (
                df['Total day calls'] +
                df['Total eve calls'] +
                df['Total night calls']
            ) / 3

            logger.info("Feature creation done")
            return df

        except Exception as e:
            raise MyException(e, sys)

    def remove_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            drop_cols = self.cfg['drop_cols']
            df = df.drop(columns=drop_cols, errors='ignore')
            logger.info("Feature removal done — dropped: %s", drop_cols)
            return df

        except Exception as e:
            raise MyException(e, sys)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_features(df)
        df = self.remove_features(df)
        return df


class PreprocessingPipeline:
    """Builds sklearn preprocessing pipeline"""

    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def build(self):
        try:
            num_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scaler', RobustScaler())
            ])

            cat_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            transformer = ColumnTransformer([
                ('num', num_pipe, self.num_cols),
                ('cat', cat_pipe, self.cat_cols)
            ], remainder='drop')

            logger.info("Preprocessing pipeline built")
            return transformer

        except Exception as e:
            raise MyException(e, sys)


class DataTransformation:
    """Orchestrates feature engineering and preprocessing"""

    def __init__(self):
        self.config = read_config_file()
        self.cfg = self.config['data_transformation']

    def run(self):
        try:
            logger.info("Data transformation started")

            # load data
            train = pd.read_csv(self.cfg['raw_train_path'])
            test = pd.read_csv(self.cfg['raw_test_path'])

            # feature engineering
            fe = FeatureEngineer(self.cfg)
            train = fe.run(train)
            test = fe.run(test)

            # split features and target
            target = self.cfg['target_col']
            X_train = train.drop(columns=[target])
            y_train = train[target]
            X_test = test.drop(columns=[target])
            y_test = test[target]

            # build and fit pipeline
            num_cols = self.cfg['num_cols']
            cat_cols = self.cfg['cat_cols']
            pipeline = PreprocessingPipeline(num_cols, cat_cols).build()

            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)

            # save pipeline
            save_object(self.cfg['preprocessor_path'], pipeline)
            logger.info("Preprocessor saved at: %s", self.cfg['preprocessor_path'])

            logger.info("Data transformation completed")
            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            raise MyException(e, sys)


if __name__ == "__main__":
    transformation = DataTransformation()
    transformation.run()