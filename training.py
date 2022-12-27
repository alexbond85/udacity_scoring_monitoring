from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")


#################Function for training the model
def train_model():
    df = pd.read_csv(dataset_csv_path)
    y_train = df["exited"].tolist()
    names = "lastmonth_activity,lastyear_activity,number_of_employees".split(",")
    column_selector = ColumnSelector(cols=names)
    # use this logistic regression for training
    log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False,
                                 fit_intercept=True,
                                 intercept_scaling=1, l1_ratio=None, max_iter=100,
                                 multi_class='ovr', n_jobs=None, penalty='l2',
                                 random_state=0, solver='liblinear', tol=0.0001,
                                 verbose=0,
                                 warm_start=False)
    pipeline = Pipeline([
        ('drop_columns', column_selector),
        ('logistic_regression', log_reg)
    ])
    # fit the logistic regression to your data
    pipeline.fit(df, y_train)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path, 'wb') as f:
        # Pickle the object and write it to the file
        pickle.dump(pipeline, f)


if __name__ == '__main__':
    train_model()
