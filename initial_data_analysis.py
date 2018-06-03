import numpy as np
import pandas as pd 
from sklearn.preprocessing import Imputer

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

class AllData:
	def __init__(self):
		""" 
		Stores data in class AllData
		"""
		self.application_train = []

	def import_data(self):
		self.application_train = pd.read_csv("~/.kaggle/competitions/home-credit-default-risk/application_train.csv")
