import os
import shutil
import json
from ingestion import list_ingested_filepath
from training import model_path
from scoring import scores_file_path

##################Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])


####################function for deployment
def store_model_into_pickle():
    from_ = [model_path, scores_file_path, list_ingested_filepath]
    to = [os.path.join(prod_deployment_path, os.path.basename(name)) for name in from_]
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for f, t in zip(from_, to):
        shutil.copy(f, t)

if __name__ == '__main__':
    store_model_into_pickle()
