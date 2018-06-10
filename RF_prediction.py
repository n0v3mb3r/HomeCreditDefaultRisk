from sklearn.ensemble import RandomForestClassifier
import initial_data_analysis as ida
import pandas as pd

# Import Training Dataset
data = ida.AllData()
data.import_train_data()
data.application_train = ida.data_imputer().fit_transform(data.application_train)
data.application_train = pd.get_dummies(data.application_train)

# Import Test Dataset
data.import_test_data()
data.application_test = ida.data_imputer().fit_transform(data.application_test)
data.application_test = pd.get_dummies(data.application_test)

# create features and target variable
features = data.application_test.columns[1:len(data.application_test.columns)]
target_variable = 'TARGET'

train = pd.get_dummies(data.application_train[features])
labels = pd.get_dummies(data.application_train[target_variable])

print('Training Random Forest Classifier')
rfc = RandomForestClassifier(n_estimators = 500, verbose = True)
rfc.fit(train, labels)

print('Generation Predictions')
test = pd.get_dummies(data.application_test[features])

predictions = pd.DataFrame(rfc.predict(test))

export = pd.concat((data.application_test['SK_ID_CURR'], predictions[1]), axis = 1)
export_headers = ["SK_ID_CURR","TARGET"]
export.columns = export_headers

export.to_csv("submission.csv", columns = export_headers, index = False)