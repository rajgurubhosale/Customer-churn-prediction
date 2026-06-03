
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

    def build(self, num_cols):
        try:
            num_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scaler', RobustScaler())
            ])

            transformer = ColumnTransformer([
                ('num', num_pipe, num_cols),
            ], remainder='drop')

            logger.info("pipeline built num:%d ", len(num_cols))
            return transformer

        except Exception as e:
            raise MyException(e, sys)
