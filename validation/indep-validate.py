from sklearn.externals import joblib
from sklearn.metrics import classification_report as clsr
import pandas as pd
import numpy as np


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = pd.read_pickle("descForVal.pkl")

df = clean_dataset(df)

X = df.as_matrix()

modelRF = joblib.load("../trainedSte3Model.pkl")

y_pred = modelRF.predict(X)

print sum(y_pred)

