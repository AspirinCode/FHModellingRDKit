#from extractSmallSet import *
#from generateDescriptors import *
import pandas as pd
import matplotlib
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
from sklearn.metrics import classification_report as clsr
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = pd.read_csv("descDFAllSet3.csv")



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


#utilities.viewTable(features_df)

# mds = MDS(n_components=1000)
# transData = mds.fit_transform(X_train[0:1000])
#
# plt.scatter(transData[0], transData[1])
# plt.show()

# print ("Actives in the training set %d" % y_train.sum())
# print ("Actives in the test set %d" % y_test.sum())
#
# clf = DummyClassifier(strategy='most_frequent',random_state=0)
# clf.fit(X_train, y_train)
#
# #print("Dummy score for Training Set: %.4f" % cross_val_score(clf, X_train, y_train, scoring='roc_auc'))
# print(cross_val_score(clf, X_test, y_test, scoring='accuracy'))
# print(cross_val_score(clf, X_test, y_test, scoring='average_precision'))


#print(cross_val_score(clf, X_test, y_test, scoring='f1'))

#model = svm.SVC()
#model.fit(X_train, y_train)

modelRF = RandomForestClassifier(n_estimators=50, max_depth=20, class_weight="balanced")
modelRF.fit(X_train, y_train)

#y_pred = modelRF.predict(X_test)

joblib.dump(modelRF, 'trainedSte3Model.pkl')

#print (y_test,y_pred)
# print "For RF"
# print("Test set score: %.4f" % modelRF.score(X_test, y_test))
# print("Training Set score: %.4f" % modelRF.score(X_train, y_train))
# print cross_val_score(modelRF, X_test, y_test, scoring='f1')
# accuracy =  cross_val_score(modelRF, X_test, y_test, scoring='accuracy')
# precision =  cross_val_score(modelRF, X_test, y_test, scoring='average_precision')
# print cross_val_score(modelRF, X_test, y_test, scoring='roc_auc')
#
# print accuracy,precision, (accuracy+precision)/2


#print sum(y_pred), sum(y_test)
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

#
# print "For SVM"
# print("model.score for Test set: %.4f" % model.score(X_test, y_test))
# print("model.score for Training set: %.4f" % model.score(X_train, y_train))
# print("F1 score for Test Set: %.4f" % cross_val_score(model, X_test, y_test, scoring='f1'))
# print("F1 score for Training Set: %.4f" % cross_val_score(model, X_train, y_train, scoring='f1'))