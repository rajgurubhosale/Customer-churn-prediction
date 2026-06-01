from sklearn.pipeline import Pipeline

def train_model_pipeline(model,tnf:Pipeline,features,target):
    model_pipeline = Pipeline([
        ('tnf',tnf),      
        ('model',model)
    ])

    model_pipeline.fit(features,target)

    return model_pipeline
