import json
import os

import requests


# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000/"


def run():
    # Call each API endpoint and store the responses
    response1 = requests.get(f'{URL}/diagnostics').text
    response2 = requests.get(f'{URL}/summarystats').text
    response3 = requests.get(f'{URL}/scoring').text
    response4 = requests.post(f'{URL}/prediction',
                              json={"filepath": "./ingesteddata/finaldata.csv"}).text

    # combine all API responses
    # responses = #combine reponses here

    # write the responses to your workspace
    with open('config.json', 'r') as f:
        config = json.load(f)

    api_returns_path = os.path.join(config['output_model_path'], "apireturns2.txt")

    with open(api_returns_path, "w") as f:
        f.write(response1)
        f.write(response2)
        f.write(response3)
        f.write(response4)
