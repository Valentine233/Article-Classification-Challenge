from graph import graph_feature
from text import text_feature
import numpy as np
from scipy.sparse import hstack
import csv
from sklearn.linear_model import LogisticRegression

g = graph_feature()
# t = text_feature()
X_train_graph = g[0]
# X_train_text = t[0]
y_train = g[1]
X_test_graph = g[2]
# X_test_text = t[2]

# X_train = hstack((X_train_graph, X_train_text))
# X_test = hstack((X_test_graph, X_test_text))

X_train = X_train_graph
X_test = X_test_graph

# Read test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Use logistic regression to classify the articles of the test set
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
