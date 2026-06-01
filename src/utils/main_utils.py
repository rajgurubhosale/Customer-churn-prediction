from src.logger import logger
import yaml
from src.exception import MyException
import sys 
import pandas as pd
from pathlib import Path
import os
import joblib

def read_config_file():
    '''read and loads the yaml file from given path
    
    return:
        config.yaml: yaml_file from config
    '''
    try:
        yaml_file_path  = 'src/config/config_yaml.yaml'
        with open(yaml_file_path,'r') as f:
            
            file = yaml.safe_load(f)
        logger.info(f'Yaml file load from {yaml_file_path} succesfully')
        return file
    except FileNotFoundError as e:
        raise MyException(e,sys)

def save_csv(path: str, df: pd.DataFrame) -> None:
    try:
        # auto create parent folder if not exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("CSV saved at: %s", path)
    except PermissionError:
        logger.error("Permission denied to save CSV at: %s", path)
        raise
    except Exception as e:
        logger.error("Failed to save CSV: %s", e)
        raise
    

def save_object(file_path, obj):
    """
    Save Python object using joblib.
    """

    try:
        
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        joblib.dump(obj, file_path)

    except Exception as e:
        raise e