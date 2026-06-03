from sklearn.pipeline import Pipeline
from src.components.module_03_data_transformation import DataTransformation
from src.utils.main_utils import read_config_file, save_object
from src.utils.mlflow_utils import setup_mlflow
from src.logger import logger
import pandas as pd
from src.exception import MyException
import sys
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

class ModelTrainer:

    def __init__(self):
        self.config = read_config_file()
        self.paths = self.config['paths']
        self.params = self.config['model_training']['params']
        self.mlflow_config = self.config['model_training']['mlflow']

    @staticmethod
    def train_model_pipeline(model, tnf: Pipeline, features, target):
        '''main model pipeline, can be used for inference'''
        model_pipeline = Pipeline([
            ('tnf', tnf),
            ('model', model)
        ])
        model_pipeline.fit(features, target.values.ravel())
        return model_pipeline

    def load_model(self):
        model = XGBClassifier(**self.params)
        return model

    def run(self):
        try:
            logger.info("model training started")
            setup_mlflow()

            X_train = pd.read_csv(self.paths['interim_train_path'])
            y_train = pd.read_csv(self.paths['interim_y_train_path'])

            num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

            logger.info("num_cols: %s", num_cols)

            transformer = DataTransformation()
            tnf_pipeline = transformer.build(num_cols )
            final_model = self.load_model()
            final_pipeline = self.train_model_pipeline(final_model, tnf_pipeline, X_train, y_train)

            mlflow.set_experiment(self.mlflow_config['experiment_name'])

            with mlflow.start_run() as run:
                # log params
                mlflow.log_params(self.params)

                # log model — only once
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path="model",
                    registered_model_name=self.mlflow_config['model_name']
                )

                # set staging alias
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(
                    self.mlflow_config['model_name']
                )[0].version

                client.set_registered_model_alias(
                    name=self.mlflow_config['model_name'],
                    alias="staging",
                    version=latest_version
                )

                logger.info("model version %s set to staging", latest_version)
                logger.info("mlflow run id: %s", run.info.run_id)

            # save locally
            save_object(self.paths['model_pipeline_path'], final_pipeline)
            logger.info("model saved at: %s", self.paths['model_pipeline_path'])

            return final_pipeline

        except Exception as e:
            raise MyException(e, sys)


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()