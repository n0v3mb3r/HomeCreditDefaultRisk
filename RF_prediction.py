from sklearn.ensemble import RandomForestClassifier
import initial_data_analysis as ida
import pandas as pd

data = ida.AllData()
data.import_data()
data.application_train = ida.data_imputer().fit_transform(data.application_train)
data.application_train = pd.get_dummies(data.application_train)

# create features and target variable
features = data.application_train.columns[1:(len(data.application_train.columns) - 1)]
target_variable = data.application_train.columns[len(data.application_train.columns) - 1]

train = pd.get_dummies(data.application_train[features])
labels = pd.get_dummies(target_variable)

rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(train, labels)