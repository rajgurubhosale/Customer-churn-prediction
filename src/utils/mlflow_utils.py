import os
import mlflow
import dagshub
from dotenv import load_dotenv



def setup_mlflow():
    load_dotenv()
    

    DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
    DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
    DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_USER_TOKEN

    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )
    
    mlflow.set_tracking_uri(os.getenv("DAGUSHUB_TRACKING_URI"))

    print("MLflow connected successfully")