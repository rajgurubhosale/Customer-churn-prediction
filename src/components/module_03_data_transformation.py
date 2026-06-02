
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.logger import logger
from src.exception import MyException


class DataTransformation:

    def __init__(self, ):
        pass

    def build(self, num_cols, cat_cols):
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
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols)
            ], remainder='drop')

            logger.info("pipeline built num:%d cat:%d", len(num_cols), len(cat_cols))
            return transformer

        except Exception as e:
            raise MyException(e, sys)
