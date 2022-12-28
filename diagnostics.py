import pickle
import time

import pandas as pd
import os
import json

import requests as requests

import ingestion
from deployment import prod_deployment_path
from training import train_model
import pkg_resources

##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")
test_data_path = os.path.join(config['test_data_path'])


##################Function to get model predictions
def model_predictions(df: pd.DataFrame):
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)

    # read the deployed model and a test dataset, calculate predictions
    return model.predict(df)  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(dataset_csv_path)
    # calculate summary statistics here
    ddf = df.describe(percentiles=[.5])
    ddf = ddf.drop(index=["min", "max", "count"])
    # ddf = ddf.append(s, ignore_index=True)
    ddf.index = ["mean", "std", "median"]
    return ddf


def missing_data():
    df = pd.read_csv(dataset_csv_path)
    df = pd.DataFrame(df.isnull().mean() * 100)
    df.columns = ["Percentage N/A"]
    return df["Percentage N/A"].tolist()


##################Function to get timings
def execution_time():
    s1 = time.perf_counter()
    ingestion.merge_multiple_dataframe()
    elapsed1 = time.perf_counter() - s1

    s2 = time.perf_counter()
    train_model()
    elapsed2 = time.perf_counter() - s2
    # calculate timing of training.py and ingestion.py
    return [elapsed1, elapsed2]


##################Function to check dependencies
def outdated_packages_list():
    with open("requirements.txt", "r") as f:
        installed_packages = [line.replace("\n", "") for line in f.readlines()]
    package_info = []
    table = []
    # Iterate through the installed packages
    for item in installed_packages:
        # Get the current version of the package
        package, version = item.split("==")
        latest_version = "N/A"

        # Use the PyPI API to get the latest version of the package
        r = requests.get(f"https://pypi.org/pypi/{package}/json")
        if r.status_code == 200:
            latest_version = r.json()['info']['version']
        table.append((package, version, latest_version))

    df = pd.DataFrame(table)
    df.columns = ["package", "current_version", "latest_version"]
    df = df[df["current_version"] != df["latest_version"]]
    return df.reset_index(drop=True)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    # df = pd.read_csv(dataset_csv_path)
    model_predictions(df)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
