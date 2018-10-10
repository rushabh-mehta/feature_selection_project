# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(data):
    X = data.drop('SalePrice',axis=1)
    y = data['SalePrice']
    random_forest_model = RandomForestClassifier()
    slf = SelectFromModel(random_forest_model)
    slf.fit_transform(X,y)
    return list(X.columns[slf.get_support()])
select_from_model(data)


