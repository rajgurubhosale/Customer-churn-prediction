import pandas as pd
import numpy as np
import sys

from src.logger import logger
from src.exception import MyException
from src.utils.main_utils import read_config_file, save_object,save_csv


class FeatureEngineer:
    """Handles feature creation and removal"""

    def __init__(self,cfg):
        self.cfg = cfg
        self.config = self.cfg['feature_engineering']
        self.paths = self.cfg['paths']
        self.global_  = self.cfg['global']


    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Creating new features")

        try:
            df['total_calls'] = (
            df['Total day calls'] +
            df['Total eve calls'] +
            df['Total night calls'] +
            df['Total intl calls']
            )
            
            df['service_call_ratio'] = df['Customer service calls'] / (df['total_calls'] + 1)
            
            df["Total_number_of_minutes"] = df["Total day minutes"] + df["Total eve minutes"] + df["Total night minutes"]
            #heavy users
            df['avg_min_per_call'] = (
            df['Total day minutes'] +
            df['Total eve minutes'] +
            df['Total night minutes']
            ) / (df['total_calls'] + 1)

            df["International_usage_rate"] = (df["Total intl minutes"]) / (df["Total_number_of_minutes"])
                
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
            df['total_charge'] = round(df['total_charge'],2)
            logger.info("Feature creation completed")
            return df

        except Exception as e:
            raise MyException(e, sys)

    def remove_multicollinear_feature(self,df,y_train, corr_threshold):
        '''Remove Multicollinearity '''
        try:

            logger.info("Checking for highly correlated features")

            drop_features = set()

            numeric_df = df.select_dtypes(include=[int, float])

            corr_matrix = numeric_df.corr().abs()

            temp_df = numeric_df.copy()
            temp_df["Churn"] = y_train.values

            target_corr = temp_df.corr().abs()["Churn"].drop("Churn")

            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            groups = []
            for feature in upper.columns:
                correlated = upper.index[
                    upper[feature] >= corr_threshold
                ].tolist()

                if correlated:
                    group = set([feature] + correlated)

                    merged = False
                    for existing_group in groups:
                        if group & existing_group:
                            existing_group.update(group)
                            merged = True
                            break

                    if not merged:
                        groups.append(group)

            for group in groups:
                group_list = list(group)

                best_feature = target_corr[group_list].idxmax()
                group_list.remove(best_feature)
                drop_features.update(set(group_list))

                drop_with_corr = {f: round(target_corr.get(f, 0), 4) for f in group_list}

                logger.info("GROUP  : %s", list(group))
                logger.info("KEEP   : %s (target corr: %.4f)", best_feature, target_corr.get(best_feature, 0))
                logger.info("DROP   : %s", drop_with_corr)

            # manual override for intl
            logger.info(
                "Applying manual override: keeping 'Total intl minutes' and dropping 'Total intl charge'"
            )

            drop_features.discard('Total intl minutes')
            drop_features.add('Total intl charge')

            drop_features_list = list(drop_features)
            df = df.drop(columns=drop_features_list, errors='ignore')

            logger.info("Final drop list: %s", drop_features_list)
            logger.info("Shape after dropping: %s", str(df.shape))

            return df, drop_features_list

        except Exception as e:
            raise MyException(e, sys)
        
    def save_data_interim(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        dropped_features_list
        ):
        try:
            logger.info("Saving feature engineering outputs")


            save_csv(
                self.paths["interim_train_path"],
                X_train
            )
            logger.info(f"Saved train to path: {self.paths['interim_train_path']}")

            save_csv(
                self.paths["interim_test_path"],
                X_test
            )
            logger.info(f"Saved X_test to path: {self.paths['interim_test_path']}")

            save_csv(
                self.paths["interim_y_train_path"],
                y_train.to_frame()
            )
            logger.info(f"Saved y_train to path: {self.paths['interim_y_train_path']}")

            save_csv(
                self.paths["interim_y_test_path"],
                y_test.to_frame()
            )
            logger.info(f"Saved y_test to path :{self.paths['interim_y_test_path']}")

            save_object(
                self.paths["dropped_features_path"],
                dropped_features_list
            )

            logger.info(
                "Interim datasets and dropped feature artifact saved successfully"
            )

            logger.info("Feature engineering outputs saved successfully")

        except Exception as e:
            logger.error(
                "Error while saving interim artifacts: %s",
                str(e)
            )
            raise MyException(e, sys)
            
        
    def fe_orchastrate(self, train, test):

        logger.info("Feature engineering started")

        target = self.global_['target_col']
        
        train = pd.read_csv(self.paths['raw_train_path'])
        test = pd.read_csv(self.paths['raw_test_path'])

        X_train = train.drop(columns=[target])
        y_train = train[target]

        X_test = test.drop(columns=[target])
        y_test = test[target]

        X_train = self.create_features(X_train)
        X_test = self.create_features(X_test)

        X_train, dropped_feature_list = self.remove_multicollinear_feature(
            X_train,
            y_train,
            self.config['corr_threshold']
        )

        X_test = X_test.drop(
            columns=dropped_feature_list,
            errors="ignore"
        )

        self.save_data_interim(
            X_train,
            X_test,
            y_train,
            y_test,
            dropped_feature_list
        )

        logger.info("Feature engineering completed")



if __name__ == "__main__":

    logger.info("Starting feature engineering")

    cfg = read_config_file()
    
    
    fe = FeatureEngineer(cfg)
    fe.fe_orchastrate()
    
