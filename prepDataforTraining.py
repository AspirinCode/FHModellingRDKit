import pandas as pd
import pickle,utilities
from sklearn.model_selection import train_test_split
from prepDataforTraining import *



# Load the data set
df = pd.read_csv()

# Remove the fields from the data set that we don't want to include in our model, there are none in this case
pass

# Replace categorical data with one-hot encoded data, there are none in this case, so the original dataframe is copied
features_df = pd.DataFrame(df)

#Remove NaN entrys that may be present

features_df.fillna(features_df.mean(), inplace=True)


# Create the X and y arrays

y = df['Outcome'].as_matrix()

# Remove the Outcome from the feature data
del features_df['Outcome']

X = features_df.as_matrix()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)




