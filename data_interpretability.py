import json
import pandas as pd

from project.columns_with_non_empty_percentage import columns_with_non_empty_percentage
categorical  = [
"WL Thoracic Diagnosis",
"TCR CITIZENSHIP",
"TCR ECMO AT LISTING",
"TCR IABP AT LISTING",
"TCR Prostacyclin Infusion",
"TCR Prostacyclin Inhalation",
"TCR Inhaled NO",
"TCR IV INOTROPES AT LISTING",
"TCR PGE AT LISTING",
"TCR OTHER MECHANISM AT LISTING",
"TCR FUNCTIONAL STATUS @ LISTING",
"TCR PRIMARY PROJECTED SOURCE PAY",
"TCR Primary Diagnosis at Listing",
"TCR DIABETES:",
"WL WAITING LIST STATUS AT TIME OF LISTING",
"TCR ETHNICITY",
"ETHNICITY CATEGORY",
"TCR LIFE SUPPORT VENTILATOR",
"region",
"LUNG PREFERENCE AT LISTING - LEFT (1=Y)",
"LUNG PREFERENCE AT LISTING - RIGHT (1=Y)",
"LUNG PREFERENCE AT LISTING - BOTH (1=Y)",
"LUNG PREFERENCE AT REMOVAL/CURRENT TIME/ TCR - LEFT (1=Y)",
"LUNG PREFERENCE AT REMOVAL/CURRENT TIME/ TCR - RIGHT (1=Y)",
"LUNG PREFERENCE AT REMOVAL/CURRENT TIME/ TCR - BOTH (1=Y)",
"Candidate Most Recent/at Removal BW4 Antigen From Waiting List",
"Candidate Most Recent/at Removal BW6 Antigen From Waiting List",
"Candidate Most Recent/at Removal C1 Antigen From Waiting List",
"Candidate Most Recent/at Removal C2 Antigen From Waiting List",
"Candidate Most Recent/at Removal DR51 Antigen From Waiting List",
"Candidate Most Recent/at Removal DR52 Antigen From Waiting List",
"Candidate Most Recent/at Removal DR53 Antigen From Waiting List",
"Candidate Most Recent/at Removal DQB1 Antigen From Waiting List",
"Candidate Most Recent/at Removal DQB2 Antigen From Waiting List",
'OPO Center - Encrypted',
'Patient on Life Support (including VAD for HR & HL)',
'TCR ABO BLOOD GROUP',
'TCR ANY PREVIOUS MALIGNANCY:',
'TCR Prior Cardiac Surgery (non-transplant)',
'TCR RECIPIENT GENDER',
'TCR STATE OF RESIDENCY',
'Transplant Center - Encrypted',
'WL Desired Organ',
'WL Listing Center - Encrypted'
]
scalar = [
"TCR WEIGHT (KG) AT TIME OF LISTING",
"TCR HEIGHT(CM) AT TIME OF LISTING",
"TCR BMI",
"TCR CITIZENSHIP",
"Days in Status 1",
"Days in Status 1A",
"Days in Status 2",
"Days in Status 1B",
"Days in Adult Status 4",
"Days in Adult Status 5",
"Days in Adult Status 2",
"Days in Adult Status 3",
"TCR WEIGHT (KG) AT TIME OF LISTING",
"TCR HEIGHT(CM) AT TIME OF LISTING",
"TCR BMI",
"TCR CITIZENSHIP",
"Days in Status 1",
"Days in Status 1A",
"Days in Status 2",
"Days in Status 1B",
"Days in Adult Status 4",
"Days in Adult Status 5",
"Days in Adult Status 2",
"Days in Adult Status 3",
"Days in Adult Status 1",
"Days in Adult Status 6",
"TOTAL DAYS ON WAITING LIST/INCLUDING INACTIVE TIME",
"WL AGE AT LISTING IN YEARS",
"Calculated Candidate Height in CM at Listing",
"Calculated Candidate Weight in KG at Listing",
"Calculated Candidate BMI at Listing",
"Calculated Candidate Height in CM at Removal/Current Time",
"Calculated Candidate Weight in KG at Removal/Current Time",
"Calculated Candidate BMI at Removal/Current Time",
"ACTUAL YEAR OF LISTING/NOT OFFSET",
"Days in Adult Status 1",
"Days in Adult Status 6",
"TOTAL DAYS ON WAITING LIST/INCLUDING INACTIVE TIME",
"WL AGE AT LISTING IN YEARS",
"Calculated Candidate Height in CM at Listing",
"Calculated Candidate Weight in KG at Listing",
"Calculated Candidate BMI at Listing",
"Calculated Candidate Height in CM at Removal/Current Time",
"Calculated Candidate Weight in KG at Removal/Current Time",
"Calculated Candidate BMI at Removal/Current Time",
"ACTUAL YEAR OF LISTING/NOT OFFSET"
]

data = pd.read_csv("project/_thoracic_data.csv")
extension = "_thoracic_data"
for x in ["INIT_DATE" , "TX_DATE"]:
    data[x] = pd.to_datetime(data[x + extension])
data["DELAY"] = (data["TX_DATE"] - data["INIT_DATE"]).dt.days
s = len(data)
cutoff_data = data[data["INIT_DATE"]>=data["INIT_DATE"].max()-pd.to_timedelta(data["DELAY"].quantile(0.8), unit='d')]
patients = cutoff_data["PT_CODE"].unique().tolist()
data = data[~data["PT_CODE"].isin(patients)]
print(f"Cutoff of {(len(data))/s*100}%")
data = data.drop(columns=[x for x in data.columns if "Unnamed" in x])
with open("project/Data/SAS Dataset 202303/Thoracic/thoracic_data.json", 'r') as file:
    # Load JSON data from the file
    metadata = json.load(file)
column_names_to_labels = metadata["column_names_to_labels"]
column_names_to_labels["DELAY"]="DELAY"
data.columns = [column_names_to_labels[x.replace("_thoracic_data", "")] for x in data.columns]
trr_cols = columns_with_non_empty_percentage(data[~data["ENCRYPTED TRR_ID"].isna()], 0.8)
wl_cols = columns_with_non_empty_percentage(data[~data["ENCRYPTED WL_ID"].isna()], 0.8)

good_columns = sorted(x for x in list(set(trr_cols).union(wl_cols)) if x)
with open('./sorted_list.json', 'w') as json_file:
    json.dump(good_columns, json_file, indent=4)

value_count_dict = dict(sorted(data[good_columns].nunique().to_dict().items(), key=lambda item: item[1]))
with open('./value_count.json', 'w') as json_file:
    json.dump(value_count_dict, json_file, indent=4)
data[good_columns].to_csv("good_data.csv")

with open('./wll_cols.json', 'w') as json_file:
    json.dump(wl_cols, json_file, indent=4)

with open('./trr_cols.json', 'w') as json_file:
    json.dump(trr_cols, json_file, indent=4)
print("Labeling")
trr_patients = set(data[~data["ENCRYPTED TRR_ID"].isna()]["ENCRYPTED PATIENT IDENTIFIER"].unique().tolist())
# data = data[~data["ENCRYPTED TRR_ID"].isna()]
data["LABEL"] = data["ENCRYPTED PATIENT IDENTIFIER"].apply(lambda x: x in trr_patients).astype(int)
print("Final data")
final_data = data[wl_cols + ["LABEL"]]#.select_dtypes(include=['number'])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree

target_column = "LABEL"
print("Patients getting trr %",final_data[target_column].sum()/len(final_data[target_column]))
X = final_data.drop(target_column, axis=1)
s = len(X)
X = X.loc[:,~X.columns.duplicated()].copy()
print(f"Droped {(len(X))/s*100}% as  duplicates" )
ignore_columns = [
    "ENCRYPTED PATIENT IDENTIFIER",
    "WL REASON FOR REMOVAL FROM THE WAITING LIST",
    "WL NUMBER OF PREVIOUS TRANSPLANTS",
    "WL MOST RECENT WAITING LIST STATUS",
    "Registration Removed for Deceased Donor Transplant",
]
X = X[scalar + [x for x in X.columns if ((x in categorical) and (x in X.columns[X.nunique() < 10]))]]
# print("Model columsn are:", X.columns)
X = X.drop([x for x in X.columns if any(c in x for c in ignore_columns)], axis=1)
with open('./model_cols.json', 'w') as json_file:
    json.dump(list(X.columns), json_file, indent=4)
X = pd.get_dummies(X, columns=[x for x in X.columns if x in categorical])

# Extract the target variable
y = final_data[target_column]

# Split the dataset into training and testing sets
print("train_test_split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = tree.DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the model
print("Fit")
model.fit(X_train, y_train)

# Make predictions on the test set
print("Predict")
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

from sklearn.tree import export_graphviz
import os

# Assuming you have already trained your RandomForestClassifier named 'model'
# Iterate through each decision tree in the forest and export it as an image
print(tree.export_text(model, feature_names=list(X_train.columns)))

# DOT data
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=list(X_train.columns),  
                                filled=True)
import graphviz
# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph.render("decision_tree_graphivz")
graph
import dtreeviz # remember to load the package

viz = dtreeviz.model(model, X, y,
                target_name="target",
                feature_names=X_train.columns,
)

v = viz.view()     # render as SVG into internal object 
v.show()                 # pop up window
v.save("/tmp/iris.svg")  # optionally save as svg