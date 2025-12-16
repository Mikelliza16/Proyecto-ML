import pickle
from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)