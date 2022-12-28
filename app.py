from flask import Flask, render_template, session, jsonify, request
import pandas as pd
from tabulate import tabulate

import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os

from diagnostics import dataframe_summary, execution_time, missing_data, \
    model_predictions, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    filepath = request.get_json()['filepath']
    df = pd.read_csv(filepath)
    res = jsonify({"predictions": [int(x) for x in model_predictions(df)]})
    return res  # add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # check the score of the deployed model
    res = jsonify({"f1": score_model()})
    return res # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    df = dataframe_summary()
    res = tabulate(df, headers='keys', tablefmt='psql')
    return f"\n<pre>\n{res}\n</pre>" # return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    t_ingest, t_train = execution_time()
    number_na = missing_data()
    d = {"ingestion_time": t_ingest, "training_time": t_train,
                   "missing_data_percentage": number_na}
    pkgs: pd.DataFrame = outdated_packages_list()
    res = tabulate(pkgs, headers='keys', tablefmt='psql') + f"\n{str(d)}"
    return f"<pre>\n{res}\n</pre>" # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
