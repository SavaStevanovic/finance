import copy
import itertools
import json
import os
import re
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg',force=True)


def plot_category_distribution(df, target_column, category_column, path):
        plt.figure(figsize=(8, 6))
        df = copy.deepcopy(df[[target_column, category_column]])
        df = df.dropna()
        if df.empty:
            return
        plt.figure()  # No figsize specified here
        chategorical = (df[category_column].nunique() <= 10) or (df[category_column].dtype.name == "object")
        if chategorical:
            sns.countplot(data=df, x=category_column, hue=target_column)
        else:
            sns.histplot(data=df, x=category_column, hue=target_column, bins=100, kde=True)
        plt.title(f'Distribution of {category_column} values with {target_column}')
        plt.xlabel(category_column)
        plt.ylabel('Count')
        plt.legend(title=target_column, labels=df[target_column].unique().tolist())
        image_dir = os.path.join(path, target_column)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, category_column + '.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')  # Adjust DPI and quality as needed
        plt.close()
        
        
# Assuming you have a DataFrame named df
# Replace 'filename.csv' with your actual filename
def extract_stats(target_columns, columns_dataset, relevant_df, filename):
    data_path = os.path.join("test_data", filename)
    os.makedirs(data_path, exist_ok=True)
    metadata = {"data_size": len(relevant_df)}
    for target in target_columns:
        metadata[target] = int(relevant_df[target].sum())
    with open(os.path.join(data_path, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=4)
    for col in columns_dataset:
        for t_col in target_columns:
            plot_category_distribution(relevant_df, t_col, col, data_path)
        trr_path = os.path.join("test_data", "transplantation")
        os.makedirs(trr_path, exist_ok=True)
        plot_category_distribution(relevant_df[relevant_df[target_columns[0]]], t_col, col, os.path.join(trr_path, filename.split(".")[0]))

def fetch_data(filename, common_cols):
    columns = ["AGE_GROUP", "GENDER", "HGT_CM_CALC", "WGT_KG_CALC", "ETHNICITY", "EDUCATION"]
    target_columns = ["TRR_ID_CODE", "WL_ID_CODE"]
    columns_dataset = [c + filename.split(".")[0] for c in columns] 
    target_columns = [c + filename.split(".")[0] for c in target_columns]
    # Load the DataFrame from the CSV file
    df = pd.read_csv(filename)
    relevant_df = df[columns_dataset + common_cols + target_columns]
    relevant_df[target_columns]= ~relevant_df[target_columns].isna()
    relevant_df["BMI" + filename.split(".")[0]] = relevant_df["WGT_KG_CALC" + filename.split(".")[0]] / (relevant_df["HGT_CM_CALC" + filename.split(".")[0]]/100) ** 2 
    columns_dataset += ["BMI" + filename.split(".")[0]]
    return target_columns,columns_dataset,relevant_df

def fetch_tx_data(filename, common_cols):
    columns = ["AGE_GROUP", "GENDER", "HGT_CM_CALC", "WGT_KG_CALC", "ETHNICITY", "EDUCATION", "TX_DATE", "INIT_DATE"]
    target_columns = ["TRR_ID_CODE", "WL_ID_CODE"]
    columns_dataset = [c + filename.split(".")[0] for c in columns] 
    target_columns = [c + filename.split(".")[0] for c in target_columns]
    # Load the DataFrame from the CSV file
    df = pd.read_csv(filename)
    relevant_df = df[columns_dataset + common_cols + target_columns]
    relevant_df[target_columns]= ~relevant_df[target_columns].isna()
    relevant_df["BMI" + filename.split(".")[0]] = relevant_df["WGT_KG_CALC" + filename.split(".")[0]] / (relevant_df["HGT_CM_CALC" + filename.split(".")[0]]/100) ** 2 
    columns_dataset += ["BMI" + filename.split(".")[0]]
    return target_columns,columns_dataset,relevant_df

def all_combinations(elements):
    all_comb = []
    for r in range(1, min(len(elements), 4)):
        all_comb.extend(list(itertools.combinations(elements, r)))
    return all_comb

filenames = ["_intestine_data.csv", "_kidpan_data.csv", "_liver_data.csv", "_thoracic_data.csv"]
file_interactions = all_combinations(filenames)
common_cols = ["PT_CODE", "DONOR_ID"]
def cols_parsing(cols):
    cols = [''.join([char for char in c if not char.islower()]) for c in cols]
    cols = [re.sub(r'_+$', '', c) for c in cols]
    return cols

for filenames in file_interactions:
    target_columns = [] 
    columns_dataset = []
    datasets = []
    for filename in filenames:
        target_column, column_dataset, relevant_df = fetch_data(filename, common_cols)
        target_columns.extend(target_column)
        columns_dataset.extend(column_dataset)
        datasets.append(relevant_df)
        
    merged_df = datasets[0]
    for df in datasets[1:]:
        merged_df = pd.merge(merged_df, df, on=common_cols, how='inner')
    filename = "_".join(filename.split(".")[0] for filename in filenames)
    extract_stats(target_columns, columns_dataset, merged_df, filename)
    target_columns = [] 
    columns_dataset = []
    datasets = []
    for filename in filenames:
        target_column, column_dataset, relevant_df = fetch_tx_data(filename, common_cols)
        target_columns.extend(target_column)
        columns_dataset.extend(column_dataset)
        datasets.append(relevant_df)
    for d in datasets:
        d.columns = cols_parsing(d.columns)
    target_columns = cols_parsing(target_columns)
    columns_dataset = cols_parsing(columns_dataset)
    concated_df = pd.concat(datasets)
    trr_df = concated_df[concated_df[[x for x in concated_df.columns if "TRR_ID_CODE" in x]].any(axis=1)]
    trr_df = trr_df[[x for x in trr_df.columns if all(col not in x for col in ["CALC", "BMI"])]]
    agg_dict_dates = {**{
        x: 'max' for x in trr_df.columns if "TX_DATE" in x
    },  **{
        x: 'min' for x in trr_df.columns if "INIT_DATE" in x
    } }
    agg_dict = {**agg_dict_dates, **{x: "max" for x in trr_df.columns if x not in list(agg_dict_dates.keys()) + ["PT_CODE"]}}
    for k in list(agg_dict_dates.keys()):
        trr_df[k] = pd.to_datetime(trr_df[k], format='%Y-%m-%d', errors='coerce')
    data = trr_df.groupby('PT_CODE').agg(agg_dict).reset_index()
    targets = []
    extension = ""
    data["DELAY" + extension] = (data["TX_DATE" + extension] - data["INIT_DATE" + extension]).dt.days
    targets.append("DELAY" + extension)
    data = data.drop(["TX_DATE" + extension, "INIT_DATE" + extension], axis=1)
    
    filename = "concated_" + "_".join(filename.split(".")[0] for filename in filenames)
    for target in targets:
        for col in [x for x in columns_dataset if all(col not in x for col in ["CALC", "TX_DATE", "INIT_DATE", "BMI"])]:
            print(filename, target, col)
            plot_category_distribution(data, col, target, os.path.join("test_data", filename))