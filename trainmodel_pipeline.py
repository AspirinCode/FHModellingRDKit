#from extractSmallSet import *
#from generateDescriptors import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# Load the data set

df = pd.read_csv("descDFAll.csv")

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



#model = svm.SVC()
#model.fit(X_train, y_train)

modelRF = RandomForestClassifier(n_estimators=100, max_depth=50)
modelRF.fit(X_train, y_train)
print "For RF"
print("Test set score: %.4f" % modelRF.score(X_test, y_test))
print("Training Set score: %.4f" % modelRF.score(X_train, y_train))
print cross_val_score(modelRF, X_test, y_test, scoring='f1')

# modelRF = RandomForestClassifier()
#
#
# param_grid = {
#     "max_depth": [2,5,10,20,50,100],
#     "n_estimators" : [10,50,100,200,500]
#     # 'n_estimators': [500, 1000, 3000],
#     # 'max_depth': [4, 6],
#     # 'min_samples_leaf': [3, 5, 9, 17],
#     # 'learning_rate': [0.1, 0.05, 0.02, 0.01],
#     # 'max_features': [1.0, 0.3, 0.1],
# }
#
# # Define the grid search we want to run. Run it with four cpus in parallel.
# gs_cv = GridSearchCV(modelRF, param_grid, n_jobs=4)
#
# # Run the grid search - on only the training data!
# gs_cv.fit(X_train, y_train)
#
# # Print the parameters that gave us the best result!
# print(gs_cv.best_params_)


# print "For SVM"
# print("Test Set Mean Absolute Error: %.4f" % model.score(X_test, y_test))
# print("Training Set Mean Absolute Error: %.4f" % model.score(X_train, y_train))

