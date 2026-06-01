from sklearn.metrics import recall_score,precision_score,confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np


def metrics_evaluation(model_pipeline,X_train,X_test,y_train_set,y_test):
     
    #metrics on train set
    train_pred = model_pipeline.predict(X_train)


    print('Train metrics')
    print(f'recall: {recall_score(y_train_set,train_pred):.3f}')
    print(f'precision:{precision_score(y_train_set,train_pred):.3f}')
    print(' ')


    # Metrics on test set
    test_pred = model_pipeline.predict(X_test)
    
    print('Test metrics')
    print(f'recall: {recall_score(y_test,test_pred):.3f}')
    print(f'precision:{precision_score(y_test,test_pred):.3f}')


    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,fmt='0.3f')

def validation_evalution(model_pipeline,metric,X_train_set,y_train_set):
    cv =  StratifiedKFold(n_splits=5,shuffle=True)
    
    cv_result = cross_val_score(model_pipeline,X_train_set,y_train_set,cv=cv,scoring=metric,)
    return np.mean(cv_result)