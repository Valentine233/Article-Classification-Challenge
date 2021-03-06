import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
from scipy.sparse import hstack


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
    class_num = unique.size
    print("Number of classes: %d" % class_num)

    # Number of authors
    authors = set()
    for i in range(len(df)):
        aut = str(df.loc[i]['authors']).split(',')
        aut = [a.strip() for a in aut]
        authors |= set(aut)
    authors = list(authors)
    n_author = len(authors)

    # token counts of authors' names
    def mapau(s):
        l = s.split(',')
        l = [i.strip() for i in l]
        idx = np.zeros(n_author)
        for i in l:
            if i in authors:
                idx[authors.index(i)] += 1
        return idx

    # Extract the abstract of each training article from the dataframe
    train_abstracts = list()
    train_titles = list()
    train_authors = list()
    for i in train_ids:
        train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
        train_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
        train_authors.append(mapau(str(df.loc[df['id'] == int(i)]['authors'].iloc[0])))

    # count_vec = CountVectorizer(decode_error='ignore', min_df=2, max_df=50, stop_words='english')
    tfidf_vec1 = TfidfVectorizer(decode_error='ignore', min_df=2, max_df=0.9, ngram_range=(2, 5), analyzer='char', stop_words='english')
    tfidf_vec3 = TfidfVectorizer(decode_error='ignore', min_df=2, max_df=0.9, ngram_range=(1, 3), analyzer='word', stop_words='english')
    tfidf_vec2 = TfidfVectorizer(decode_error='ignore', min_df=2, max_df=0.9, ngram_range=(2, 5), analyzer='char', stop_words='english')
    tfidf_vec4 = TfidfVectorizer(decode_error='ignore', min_df=2, max_df=0.9, ngram_range=(1, 3), analyzer='word', stop_words='english')
    TrainAbstracts = tfidf_vec1.fit_transform(train_abstracts) # TF-IDF features of abstracts with ’char’ analyzer and n-gram from 2 to 5
    wTrainAbstracts = tfidf_vec3.fit_transform(train_abstracts) # TF-IDF features of abstracts with ’word’ analyzer and n-gram from 1 to 3
    TrainTitles = tfidf_vec2.fit_transform(train_titles) # TF-IDF features of titles with ’char’ analyzer and n-gram from 2 to 5
    wTrainTitles = tfidf_vec4.fit_transform(train_titles) # TF-IDF features of titles with ’word’ analyzer and n-gram from 1 to 3
    TrainAuthors = sparse.csr_matrix(train_authors) # Token counts of authors’ name
    X_train = hstack((TrainAbstracts, TrainTitles, wTrainAbstracts, wTrainTitles, TrainAuthors))

    # Read test data
    test_ids = list()
    with open('test.csv', 'r') as f:
        next(f)
        for line in f:
            test_ids.append(line[:-2])

    # Extract the abstract of each test article from the dataframe
    n_test = len(test_ids)
    test_abstracts = list()
    test_titles = list()
    test_authors = list()
    for i in test_ids:
        test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
        test_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
        test_authors.append(mapau(str(df.loc[df['id'] == int(i)]['authors'].iloc[0])))

    # Create the test matrix following the same approach as in the case of the training matrix
    TestAbstracts = tfidf_vec1.transform(test_abstracts)
    wTestAbstracts = tfidf_vec3.transform(test_abstracts)
    TestTitles = tfidf_vec2.transform(test_titles)
    wTestTitles = tfidf_vec4.transform(test_titles)
    TestAuthors = sparse.csr_matrix(test_authors)
    X_test = hstack((TestAbstracts, TestTitles, wTestAbstracts, wTestTitles, TestAuthors))

    print("Train matrix dimensionality: (%d, %d)" % (X_train.shape[0], X_train.shape[1]))
    print("Test matrix dimensionality: (%d, %d)" % (X_test.shape[0], X_test.shape[1]))
    return X_train, y_train, X_test
