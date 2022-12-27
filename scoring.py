import pandas as pd
import pickle
import os
from sklearn import metrics
import json
from training import model_path

#################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
scores_file_path = os.path.join(config['output_model_path'], "latestscore.txt")


#################Function for model scoring
def score_model():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(test_data_path)
    y_true = df["exited"]
    y_pred = model.predict(df)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    with open(scores_file_path, "a") as f:
        f.write(str(f1))
        f.write("\n")


if __name__ == '__main__':
    score_model()
