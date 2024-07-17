import json
import pandas as pd


data = pd.read_csv("project/_thoracic_data.csv")
data = data.drop(columns=[x for x in data.columns if "Unnamed" in x])
with open("project/Data/SAS Dataset 202303/Thoracic/thoracic_data.json", 'r') as file:
    # Load JSON data from the file
    metadata = json.load(file)
column_names_to_labels = metadata["column_names_to_labels"]
data.columns = [column_names_to_labels[x.replace("_thoracic_data", "")] for x in data.columns]