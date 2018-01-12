from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import classification_report as clsr
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = pd.read_csv("descDFAll.csv")



# Remove the fields from the data set that we don't want to include in our model, there are none in this case
pass

# Replace categorical data with one-hot encoded data, there are none in this case, so the original dataframe is copied
features_df = df

#Remove NaN entrys that may be present

features_df = clean_dataset(features_df)
# Create the X and y arrays

y = features_df['Outcome'].as_matrix().astype(np.float)

print len(features_df.columns)

# Remove the Outcome from the feature data
del features_df['Outcome']

print len(features_df.columns)
X = features_df.as_matrix().astype(np.float)



# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


modelRF = joblib.load("trainedSte3Model.pkl")

y_pred = modelRF.predict(X_test)
y_pred = modelRF.predict(X_train)

print y_pred

print clsr(y_test, y_pred)