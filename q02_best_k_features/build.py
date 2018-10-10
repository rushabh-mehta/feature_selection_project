# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(data,k=20):
    X = data.drop('SalePrice',axis=1)
    y = data['SalePrice']
    feature_names = X.columns
    fs = SelectPercentile(f_regression, percentile=k)
    X_new = fs.fit_transform(X, y)
    features_by_scores = [feature_names[i] for i in np.argsort(fs.scores_)[::-1]]
    return features_by_scores[:7]
percentile_k_features(data,20)



