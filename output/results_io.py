import pandas as pd
from sklearn.metrics import recall_score,precision_score
import os
RESULTS_PATH = r'output/results.csv'

def save_results(model_pipeline,model_name,X_train,X_test,y_train,y_test):

    train_pred = model_pipeline.predict(X_train)
    test_pred = model_pipeline.predict(X_test)

    #train
    train_recall = recall_score(y_train,train_pred)
    train_precision = precision_score(y_train,train_pred)

    # test
    test_recall = recall_score(y_test,test_pred)
    test_precision = precision_score(y_test,test_pred)

    scores = pd.DataFrame({
        'model':model_name,
        'recall_test':[test_recall],
        'precision_test':[test_precision],
        'recall_train':[train_recall],
        'precision_train':[train_precision]
    })

    if os.path.exists(RESULTS_PATH) and os.path.getsize(RESULTS_PATH) > 0:
        df = pd.read_csv(RESULTS_PATH)      
        new_df = pd.concat([df,scores],ignore_index=True)
        new_df.to_csv(RESULTS_PATH,index=False)
    else:
        new_df = pd.DataFrame(scores)
        new_df.to_csv(RESULTS_PATH,index=False)


def load_results():
    """Loads the results CSV if it exists and is not empty, otherwise returns an empty DataFrame."""
    if os.path.exists(RESULTS_PATH) and os.path.getsize(RESULTS_PATH) > 0:
        return pd.read_csv(RESULTS_PATH)
    else:
        print('File is empty')


    





