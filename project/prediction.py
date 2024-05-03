import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
dataset = "_liver_data"
data_path = f'processed{dataset}.csv'
# Assuming df is your dataframe
df = pd.read_csv(data_path)
# Drop the target variable from the features
target_column = 'PSTATUS' + dataset
X = df.drop(target_column, axis=1)
ignore_columns = [
    "COD", 
    "PX_STAT_liver_data", 
    "GSTATUS_liver_data", 
    "TX_YEAR_liver_data", 
    "Unnamed",
    "LISTYR_liver_data",
    "PT_CODE"
]
X = X.drop([x for x in X.columns if any(c in x for c in ignore_columns)], axis=1)

# Extract the target variable
y = df[target_column]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = tree.DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
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