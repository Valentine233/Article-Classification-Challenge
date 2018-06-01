import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer


def text_feature():
    # Load data about each article in a dataframe
    df = pd.read_csv("node_information.csv")
    print(df.head())

    # Read training data
    train_ids = list()
    y_train = list()
    with open('train.csv', 'r') as f:
        next(f)
        for line in f:
            t = line.split(',')
            train_ids.append(t[0])
            y_train.append(t[1][:-1])

    n_train = len(train_ids)
    unique = np.unique(y_train)
    print("\nNumber of classes: ", unique.size)

    # Extract the abstract of each training article from the dataframe
    train_abstracts = list()
    for i in train_ids:
        train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])

    # Create the training matrix. Each row corresponds to an article
    # and each column to a word present in at least 2 webpages and at
    # most 50 articles. The value of each entry in a row is equal to
    # the frequency of that word in the corresponding article
    vec = CountVectorizer(decode_error='ignore', min_df=2, max_df=50, stop_words='english')
    X_train = vec.fit_transform(train_abstracts)

    # Read test data
    test_ids = list()
    with open('test.csv', 'r') as f:
        next(f)
        for line in f:
            test_ids.append(line[:-2])

    # Extract the abstract of each test article from the dataframe
    n_test = len(test_ids)
    test_abstracts = list()
    for i in test_ids:
        test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])

    # Create the test matrix following the same approach as in the case of the training matrix
    X_test = vec.transform(test_abstracts)

    print("\nTrain matrix dimensionality: ", X_train.shape)
    print("Test matrix dimensionality: ", X_test.shape)
    return X_train, y_train, X_test
