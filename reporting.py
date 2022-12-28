import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

confusion_matrix_path = os.path.join(config['output_model_path'], "confusionmatrix.png")
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
trained_model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)
    print(model)
    df = pd.read_csv(test_data_path)
    y_true = df["exited"]
    y_pred = model.predict(df)
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cbar=False)

    # Add labels to the plot
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.savefig(confusion_matrix_path)

    #write the confusion matrix to the workspace


if __name__ == '__main__':
    score_model()


if __name__ == '__main__':
    score_model()
