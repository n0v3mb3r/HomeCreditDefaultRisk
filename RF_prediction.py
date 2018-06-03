from sklearn.ensemble import RandomForestClassifier
import initial_data_analysis as ida
import pandas as pd

# Import Training Dataset
data = ida.AllData()
data.import_train_data()
data.application_train = ida.data_imputer().fit_transform(data.application_train)
data.application_train = pd.get_dummies(data.application_train)

# create features and target variable
features = data.application_train.columns[1:(len(data.application_train.columns) - 1)]
target_variable = data.application_train.columns[len(data.application_train.columns) - 1]

train = pd.get_dummies(data.application_train[features])
labels = pd.get_dummies(data.application_train[target_variable])

print('Training Random Forest Classifier')
rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(train, labels)

print('Generation Predictions')

# Import Test Dataset
data.import_test_data()
data.application_test = ida.data_imputer().fit_transform(data.application_test)
data.application_test = pd.get_dummies(data.application_test)

test = pd.get_dummies(data.application_test[features])

predictions = pd.DataFrame(rfc.predict(test))

export = pd.concat(test["SK_ID_CURR"], predictions, axis = 1)
export_headers = ["SK_ID_CURR","TARGET"]
export.columns = export_headers

export.to_csv("submission.csv", columns = export_headers, index = False)