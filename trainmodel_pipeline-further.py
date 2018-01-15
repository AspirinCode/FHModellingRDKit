import pandas as pd
import utilities
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report as clsr


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = pd.read_csv("descDFAll.csv")

df = df.sample(frac=0.1)

# Remove the fields from the data set that we don't want to include in our model, there are none in this case
pass

# Replace categorical data with one-hot encoded data, there are none in this case, so the original dataframe is copied
features_df = df

# Remove NaN entrys that may be present

features_df = clean_dataset(features_df)
# Create the X and y arrays

y = features_df['Outcome'].as_matrix()

print len(features_df.columns)

# Remove the Outcome from the feature data, abd the index parameter that got generated while reading in the csv
features_df.drop('Outcome', axis=1, inplace=True)
features_df.drop(features_df.columns[[0]], axis=1, inplace=True)

print len(features_df.columns)

# utilities.viewTable(features_df)

X = features_df.as_matrix().astype(np.float)

skf = StratifiedKFold(n_splits=5, shuffle= True)

skf.get_n_splits(X, y)

# Split the data set in a training set (70%) and a test set (30%)

for train_index, test_index, in skf.split(X, y):

    X_train, X_test, = X[train_index], X[test_index]

    y_train, y_test =  y[train_index], y[test_index]

    modelRF = RandomForestClassifier(n_estimators=2000, max_depth=5, class_weight="balanced", n_jobs=16)
    modelRF.fit(X_train, y_train)

    y_pred = modelRF.predict(X_test)
    y_predTr = modelRF.predict(X_train)

    print clsr(y_test, y_pred)
    print clsr(y_train, y_predTr)
    print cross_val_score(modelRF, X_test, y_test, scoring='f1')
    print cohen_kappa_score(y_pred, y_test)
