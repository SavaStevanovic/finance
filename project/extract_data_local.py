import os
import pandas as pd
import pyreadstat
import json
from datetime import datetime
import csv
file_path = 'output_file.json'

df, meta = pyreadstat.read_sas7bdat('project/Data/SAS Dataset 202303/Thoracic/thoracic_data.sas7bdat', catalog_file='project/Data/SAS Dataset 202303/Thoracic/formats.sas7bcat', formats_as_category=True, formats_as_ordered_category=False)


def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

# Dump the dictionary into the file using the custom serialization function
with open(file_path, 'w') as json_file:
    json.dump(meta.__dict__, json_file, default=serialize_datetime, indent=4)
df


def process_sas_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.sas7bdat'):
                sas_file_path = os.path.join(root, file)
                catalog_file_path = os.path.join(root, 'formats.sas7bcat')
                try:
                    df, meta = pyreadstat.read_sas7bdat(sas_file_path, catalog_file=catalog_file_path, formats_as_category=True, formats_as_ordered_category=False)
                    # Save DataFrame to CSV
                    csv_file_path = os.path.splitext(sas_file_path)[0] + '.csv'
                    df.to_csv(csv_file_path, index=False)
                    
                    # Convert meta object to a JSON serializable dictionary
                    meta_dict = meta.__dict__  # Convert meta object to dictionary
                    
                    # Save meta data to JSON
                    json_file_path = os.path.splitext(sas_file_path)[0] + '.json'
                    with open(json_file_path, 'w') as json_file:
                        json.dump(meta_dict, json_file, indent=4, default=str)
                        
                    print(f"Processed {sas_file_path}")
                except Exception as e:
                    print(f"Error processing {sas_file_path}: {e}")
                    
process_sas_files("/mnt/FastData/Data/UNOS Data")      

def find_csv_with_columns(directory, reference_columns):

    # List to store paths of CSV files containing all columns
    matching_files = []

    # Recursively iterate through all directories and files
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                csv_path = os.path.join(root, filename)
                # Check if all columns in the reference CSV file are present in the current CSV file
                with open(csv_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    csv_columns = next(csv_reader)
                    if set(reference_columns).issubset(set(csv_columns)):
                        matching_files.append(csv_path)

    return matching_files


def merge_dataframes(csv_paths, cols):
    # List to store DataFrames read from CSVs
    dfs = []
    
    # Read each CSV into a DataFrame and store in the 'dfs' list
    for path in csv_paths:
        # if "intestine" not in path:
        df = pd.read_csv(path)
        suffix = "_" + path.split("/")[-1].rstrip(".csv") 
        df.columns = [c + suffix if c not in cols else c for c in df.columns]
        # df = df[~df[cols].isna().any(axis=1)]
        print(df.shape, suffix)
        df.to_csv(f"{suffix}.csv")
        dfs.append(df)
    
    # Merge DataFrames based on the specified columns
    merged_df = dfs[0]  # Initialize merged DataFrame with the first DataFrame
    
    for i, df in enumerate(dfs[1:]):
        # merge_cols = list(set(merged_df.columns).intersection(df.columns))
        # for col in merge_cols:
        #     merged_df[col] = merged_df[col].astype('object')
        #     df[col] = df[col].astype('object')
        merged_df = pd.merge(merged_df, df, on=cols, how='inner')
        print(merged_df.shape)
    
    return merged_df
cols = ["PT_CODE", "DONOR_ID"]
df_paths = find_csv_with_columns("/mnt/FastData/Data/UNOS Data", cols)
print(df_paths)
# exit()
merge_dataframes(df_paths, cols).to_csv("relevant_data.csv")