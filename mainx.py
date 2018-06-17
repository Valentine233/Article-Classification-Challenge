from graph import graph_feature
from text import text_feature
import numpy as np
from scipy.sparse import hstack
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

t = text_feature()
g = graph_feature()

X_train_graph = g[0]
X_train_text = t[0]
y_train = g[1]
X_test_graph = g[2]
X_test_text = t[2]

X_train = hstack((X_train_graph, X_train_text))
X_test = hstack((X_test_graph, X_test_text))

# X_train = X_train_graph
# X_test = X_test_graph

# Use voting to classify the articles of the test set
clf1 = LogisticRegression(random_state=1)
clf2 = xgb.XGBClassifier()
eclf = VotingClassifier(estimators=[('lr', clf1), ('xgb', clf2)], voting='soft')

# for clf, label in zip([clf1, clf2, eclf], ['Logistic Regression', 'Xgboost', 'Ensemble']):
#     scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='log_loss')
#     print("Log loss: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Read test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

eclf.fit(X_train, y_train)
y_pred = eclf.predict_proba(X_test)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = eclf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
