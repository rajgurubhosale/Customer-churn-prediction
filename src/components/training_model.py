from sklearn.pipeline import Pipeline
from src.components.module_03_data_transformation import  DataTransformation
from src.utils.main_utils import read_config_file,save_object
from src.logger import logger
import pandas as pd
from src.exception import MyException
import sys
from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    
    def __init__(self):
        self.config = read_config_file()
        self.paths = self.config['paths']


    def train_model_pipeline(model,tnf:Pipeline,features,target):
        '''
        main model pipeline 
        can be used for inference'''
        model_pipeline = Pipeline([
            ('tnf',tnf),      
            ('model',model)
        ])

        model_pipeline.fit(features,target)

        return model_pipeline

    def load_model():
        final_model = LogisticRegression()
        
        return final_model


    def run(self):
        try:    
            logger.info("data transformation started")

            X_train = pd.read_csv(self.paths['interim_train_path'])
            y_train = pd.read_csv(self.paths['interim_y_train_path'])

            num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            logger.info("num_cols: %s", num_cols)
            logger.info("cat_cols: %s", cat_cols)

            transformer = DataTransformation()
            tnf_pipeline = transformer(num_cols, cat_cols)
            final_model = self.load_model()
            
            final_pipeline = self.train_model_pipeline(final_model,tnf_pipeline,X_train,y_train)
            
            save_object(self.paths['model_pipeline_path'],final_pipeline)
            
        
        except Exception as e:
            raise MyException(e, sys)
