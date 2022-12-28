import pandas as pd
import os
import json

#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
csv_merged_filepath = os.path.join(output_folder_path, "finaldata.csv")
list_ingested_filepath = os.path.join(output_folder_path, "ingestedfiles.txt")


#############Function for data ingestion
def merge_multiple_dataframe():
    csv_files = []
    for filename in sorted(os.listdir(input_folder_path)):
        if filename.endswith("csv"):
            csv_files.append(filename)
    csv_files = [os.path.join(input_folder_path, filename) for filename in csv_files]
    # Read each CSV file into a separate DataFrame, and store them all in a list
    df_list = []
    for file in csv_files:
        df_list.append(pd.read_csv(file))

    # Concatenate all the DataFrames into a single DataFrame
    df: pd.DataFrame = (
        pd.concat(df_list, axis=0, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df.to_csv(csv_merged_filepath, index=False)
    with open(list_ingested_filepath, 'w', encoding='utf-8', newline='') as f:
        f.writelines("\n".join([os.path.basename(x) for x in csv_files]))
        f.write("\n")
    # check for datasets, compile them together, and write to an output file


if __name__ == '__main__':
    merge_multiple_dataframe()
