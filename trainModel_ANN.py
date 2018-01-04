from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from prepDataforTraining import *
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(20, 10), random_state=1)

model.fit(X_train, y_train)


# # Fit regression model
# model = ensemble.GradientBoostingRegressor(
#     n_estimators=1000,
#     learning_rate=0.1,
#     max_depth=6,
#     min_samples_leaf=9,
#     max_features=0.1,
#     loss='huber'
# )
# model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

print cross_val_score(model, X_test, y_test, scoring='roc_auc')

# # Find the error rate on the training set
# mse = mean_absolute_error(y_train, model.predict(X_train))
# print("Training Set Mean Absolute Error: %.4f" % mse)
#
# # Find the error rate on the test set
# mse = mean_absolute_error(y_test, model.predict(X_test))
# print("Test Set Mean Absolute Error: %.4f" % mse)

