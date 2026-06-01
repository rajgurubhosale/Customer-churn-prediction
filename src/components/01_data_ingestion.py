import pandas as pd
from src.logger import logger
from src.exception import MyException
import sys
from src.utils.main_utils import read_config_file,save_csv
import os
import psycopg
from dotenv import load_dotenv

# load data
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import os
print("HOST:", os.getenv("DB_HOST"))
print("PASSWORD:", os.getenv("DB_PASSWORD"))
print("DB:", os.getenv("DB_NAME"))
class DataIngestion:
    

    def __init__(self):
        self.config = read_config_file()
        self.cfg    = self.config['data_ingestion']
        load_dotenv()
        
    def get_db_connection(self):
        """Create PostgreSQL connection"""
        try:
            conn = psycopg.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD")
            )
            logger.info("PostgreSQL connection successful")
            return conn
        except Exception as e:
            raise MyException(e, sys)
            
    def load_data(self, table_name) -> pd.DataFrame:
        """Load data from PostgreSQL table"""
        try:
            logger.debug("Loading data from table: %s", table_name)

            with self.get_db_connection() as conn:
                df = pd.read_sql(
                    f"SELECT * FROM {table_name}",
                    conn
                )

            logger.info(f"Data loaded :dname: {table_name}— shape: %s", str(df.shape))
            return df

        except Exception as e:
            raise MyException(e, sys)
        
        
    # preprocessing
    def preprocess_data(self,df:pd.DataFrame) -> pd.DataFrame:
        '''
        Perform basic preprocesing for data:
        params:
            df: dataframe where changes will be done
            target_col:str-> target column name
            
        '''
        try:

            target_col  = self.cfg['target_col']
            binary_cols = self.cfg['binary_cols_to_map']
            

            for col in binary_cols:
                if col in df.columns :
                    df[col] = df[col].str.strip().map({"No": 0, "Yes": 1})
                    
            logger.debug("Encoded binary columns: %s", binary_cols)
            
            
            # convert target variable into 0,1
            df[target_col] = df[target_col].map({"No": 0, "Yes": 1})
            logger.debug("Encoded target column: %s", target_col)

          
            
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                logger.warning("Null values found after preprocessing: %d", null_count)
            
            logger.info("Preprocessing done — shape: %s", str(df.shape))
            return df
            
        
        except Exception as e:
            raise MyException(e,sys)

    def save_data(self, df: pd.DataFrame, save_path: str) -> None:
        """Save processed data to artifacts"""
        try:
            save_csv(save_path, df)
            logger.info("Data saved at: %s", save_path)
        except Exception as e:
            raise MyException(e, sys)

        
    def orchestrate(self):
        """Run full data ingestion pipeline"""
        try:
            logger.info("Data ingestion started")

            train = self.load_data(self.cfg['train_db_name'])
            test = self.load_data(self.cfg['test_db_name'])

            train = self.preprocess_data(train)
            test = self.preprocess_data(test)

            self.save_data(train, self.cfg['raw_train_path'])
            self.save_data(test, self.cfg['raw_test_path'])

            logger.info("Data ingestion completed successfully")
            return train, test

        except Exception as e:
            raise MyException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.orchestrate()
