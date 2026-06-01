from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocessing_pipeline(num_cols,cat_cols):

    #null values arent present in the train data but it is for future ref imputation is used

    num_pipe = Pipeline([ 
        ('impute',SimpleImputer(strategy='mean')),
        ('scaler',RobustScaler())
    ])
    cat_pipe = Pipeline([
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('ohe',OneHotEncoder()),
    ])

    transformer = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ],remainder='drop')

    return transformer
    




