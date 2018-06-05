# impute objects with _unknown_
# impute everything else with mean

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

class data_imputer(TransformerMixin):

    def __init__(self):
        """
        Impute objects with most common record
        Impute everything else with mean
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

data = [
    ['a', 1, 2],
    ['b', 1, 1],
    ['b', 2, 2],
    [np.nan, np.nan, np.nan]
]

X = pd.DataFrame(data)
xt = data_imputer().fit_transform(X)

print('before...')
print(X)
print('after...')
print(xt)