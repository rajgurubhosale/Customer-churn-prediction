import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, accuracy_score, recall_score,f1_score,roc_auc_score
from src.utils.main_utils import read_config_file
from src.utils.mlflow_utils import setup_mlflow
from src.logger import logger
from src.exception import MyException


class ModelEvaluation:

    def __init__(self):
        self.config = read_config_file()
        self.paths = self.config['paths']
        self.mlflow_config = self.config['model_training']['mlflow']

    def load_model(self):
        model_uri = f"models:/{self.mlflow_config['model_name']}@staging"
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("model loaded from mlflow: %s", model_uri)
        return pipeline

    def evaluate(self, pipeline, X_train, y_train, X_test, y_test):
        
        # train metrics
        y_train_pred = pipeline.predict(X_train)
        
        # test metrics
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]  # for AUC

        metrics = {
            # train
            "train_precision": precision_score(y_train, y_train_pred),
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_recall": recall_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),

            # test
            "test_precision": precision_score(y_test, y_test_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_auc": roc_auc_score(y_test, y_test_prob)
        }

        for k, v in metrics.items():
            logger.info("%s: %f", k, v)

        return metrics
    
    def run(self):
        try:
            setup_mlflow()
            mlflow.set_experiment(self.mlflow_config['experiment_name'])

            X_train = pd.read_csv(self.paths['interim_train_path'])
            y_train = pd.read_csv(self.paths['interim_y_train_path'])
            X_test = pd.read_csv(self.paths['interim_test_path'])
            y_test = pd.read_csv(self.paths['interim_y_test_path'])

            pipeline = self.load_model()

            metrics = self.evaluate(
                pipeline,
                X_train, y_train.values.ravel(),
                X_test, y_test.values.ravel()
            )

            # get the run_id from the registered model
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version_by_alias(
                self.mlflow_config['model_name'], "staging"
            )
            run_id = model_version.run_id  # ← same run as training

            # log metrics to SAME run
            with mlflow.start_run(run_id=run_id):  # ← pass run_id here
                mlflow.log_metrics(metrics)
                logger.info("metrics logged to training run: %s", run_id)

        except Exception as e:
            raise MyException(e, sys)


if __name__ == '__main__':
    evaluator = ModelEvaluation()
    evaluator.run()