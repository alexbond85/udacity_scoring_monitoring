import json
import os

import training
import scoring
import deployment
import ingestion
import reporting
import apicalls

with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']


ingestedfiles_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
model_score_prod_file = os.path.join(prod_deployment_path, "latestscore.txt")
model_score_practice_file = os.path.join(output_model_path, "latestscore.txt")
ingested_data = os.path.join(output_folder_path, "finaldata.csv")

##################Check and read new data
# first, read ingestedfiles.txt
# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

def is_new_data_ingested() -> bool:
    with open(ingestedfiles_path, "r") as f:
        ingested_files = [l.replace("\n", "") for l in f.readlines()]
    input_csv_files = []
    for filename in sorted(os.listdir(input_folder_path)):
        if filename.endswith("csv"):
            input_csv_files.append(filename)
    if not all(f in input_csv_files for f in ingested_files):
        print("previously ingested:", ingested_files, "new files:", input_csv_files)
        print("new data detected, starting ingestion")
        ingestion.merge_multiple_dataframe()
        return True
    return False




# if check_and_read_new_data():
#     training.train_model()
#     scoring.score_model()
#     if check_model_drift():
#         deployment.store_model_into_pickle()


def is_model_drift():
    with open(model_score_prod_file, "r") as f:
        latest_prod_score_str = float(f.read())
    training.train_model()
    with open(model_score_practice_file, "r") as f:
        latest_trained_model_score = float(f.read())
    return latest_prod_score_str < latest_trained_model_score


##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model


if is_new_data_ingested(): # and is_model_drift():
    training.train_model()
    scoring.score_model()
    deployment.store_model_into_pickle()
    reporting.run()
    apicalls.run()



