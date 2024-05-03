import json
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def compute_mutual_information(dataframe, target_column):
    # Select numeric columns excluding the target column
    numeric_columns = [col for col in dataframe.select_dtypes(include=['number']).columns if col != target_column]
    # Compute mutual information for each column relative to the target column
    mi_scores = {}
    for col in numeric_columns:
        mi_scores[col] = mutual_info_regression(dataframe[[col]], dataframe[target_column])[0]
    return mi_scores


def compute_correlations_with_pstatus(dataframe, column_name):
    # Select numeric columns excluding those starting with 'Unnamed'
    numeric_columns = [col for col in dataframe.select_dtypes(include=['number']).columns]
    correlations = dataframe[numeric_columns].corr(method='pearson', min_periods=1)[column_name].drop(column_name)
    return correlations

# Assuming you have a DataFrame named df
# Replace 'filename.csv' with your actual filename
filename = '_liver_data.csv'

# Load the DataFrame from the CSV file
df = pd.read_csv(filename).fillna(0)
df.dropna(axis=1, how='all', inplace=True)
columns_with_dash = [col for col in df.columns if df[col].astype(str).str.contains('-').any()]
# columns_to_exclude= columns_with_dash
df = df[[x for x in df.columns if ((x not in columns_with_dash) and (not x.startswith('Unnamed')) and ("ID" not in x) )]]
df_orig = df
print(df.shape)
df.replace({True: 1, False: 0}, inplace=True)
categorical_columns = df.select_dtypes(include=['object', 'category'])
columns_to_encode = [col for col in categorical_columns if df[col].nunique() < 10]
string_columns = [col for col in df.columns if any(isinstance(val, str) for val in df[col])]

df = pd.get_dummies(df, columns=sorted(set(columns_to_encode+ list(string_columns))))
print(df.shape)

# Specify the target column
target_column = "PSTATUS" + filename.split(".")[0]

sig_data = df.sample(frac=0.1, random_state=42)
# Compute mutual information scores
mi_scores = compute_mutual_information(sig_data, target_column)

# Sort the scores in descending order
sorted_mi_scores = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

# Display the significance of each column relative to the target column
print("Significance of each column relative to", target_column, ":")
columns_to_exclude = []
for col, score in sorted_mi_scores:
    print(col, ":", score)
    if score==0:
        columns_to_exclude.append(col)

print()
correlations = compute_correlations_with_pstatus(sig_data, target_column)
sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
print("Correlation of each column relative to", target_column, ":")
# Compute correlations with target column
for col, score in sorted_correlations:
    print(col, ":", score)
    if score==0:
        columns_to_exclude.append(col)

occ = {c: int(df_orig[c].nunique()) for c in df_orig.columns}
occ = dict(sorted(occ.items(), key=lambda item: item[1]))
print(json.dumps(occ, indent=4))

df[[c for c in df.columns if c not in columns_to_exclude]].to_csv("processed" + filename)