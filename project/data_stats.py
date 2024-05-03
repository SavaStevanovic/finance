import copy
import json
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg',force=True)

# Assuming you have a DataFrame named df
# Replace 'filename.csv' with your actual filename
filename = '_liver_data.csv'
common_cols = ["PT_CODE"]
columns = ["AGE_GROUP", "GENDER", "HGT_CM_CALC", "WGT_KG_CALC", "ETHNICITY", "EDUCATION"]
target_columns = ["TRR_ID_CODE", "WL_ID_CODE"]
columns_dataset = [c + filename.split(".")[0] for c in columns] 
target_columns = [c + filename.split(".")[0] for c in target_columns]
# Load the DataFrame from the CSV file
df = pd.read_csv(filename)
relevant_df = df[columns_dataset + common_cols + target_columns]
relevant_df[target_columns]= relevant_df[target_columns].isna()
relevant_df["BMI" + filename.split(".")[0]] = relevant_df["WGT_KG_CALC" + filename.split(".")[0]] / (relevant_df["HGT_CM_CALC" + filename.split(".")[0]]/100) ** 2 
columns_dataset += ["BMI" + filename.split(".")[0]]
def plot_category_distribution(df, target_column, category_column):
    plt.figure(figsize=(8, 6))
    df = copy.deepcopy(df[[target_column, category_column]])
    df = df.dropna()
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
    plt.savefig("test_data/" + category_column + "_" + target_column + '.png', dpi=150, bbox_inches='tight')  # Adjust DPI and quality as needed
    plt.close()
    
for t_col in target_columns:
    for col in columns_dataset:
        plot_category_distribution(relevant_df, t_col, col)