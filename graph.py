import networkx as nx
import numpy as np
import csv

def graph_feature():
    # Create a directed graph
    G = nx.read_edgelist('Cit-HepTh.txt', delimiter='\t', create_using=nx.DiGraph())
    H = nx.Graph(G)

    print("Nodes: %d" % G.number_of_nodes())
    print("Edges: %d" % G.number_of_edges())

    # Read training data
    train_ids = list()
    y_train = list()
    with open('train.csv', 'r') as f:
        next(f)
        for line in f:
            t = line.split(',')
            train_ids.append(t[0])
            y_train.append(t[1][:-1])

    # number of classes
    n_train = len(train_ids)
    unique = np.unique(y_train)
    y_set = {}
    for i, y in enumerate(unique):
        y_set[y] = i
    class_num = unique.size
    print("Number of classes: %d" % class_num)

    # Create the training matrix. Each row corresponds to an article.
    # Use the following 32 features for each article:
    # (1) out-degree of node
    # (2) in-degree of node
    # (3) average degree of neighbors of node
    # (4) clustering coefficient of node
    # (5)-(32) class of node's successors
    # (33)-(60) class of node's predecessors

    avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_ids)
    cluster = nx.clustering(H, nodes=train_ids)
    succs_class = {}
    preds_class = {}
    for id in train_ids:
        succ_class = np.zeros(28)
        pred_class = np.zeros(28)
        for neig in G.neighbors(id):
            if neig in train_ids:
                succ_class[y_set[y_train[train_ids.index(neig)]]] += 1
        for neig in G.predecessors(id):
            if neig in train_ids:
                pred_class[y_set[y_train[train_ids.index(neig)]]] += 1
        succs_class[id] = succ_class
        preds_class[id] = pred_class

    X_train = np.zeros((n_train, 4+class_num*2))
    for i in range(n_train):
    	X_train[i,0] = G.out_degree(train_ids[i])
    	X_train[i,1] = G.in_degree(train_ids[i])
    	X_train[i,2] = avg_neig_deg[train_ids[i]]
        X_train[i,3] = cluster[train_ids[i]]
        for a in range(class_num):
            X_train[i,4+a] = succs_class[train_ids[i]][a]
        for a in range(class_num):
            X_train[i,4+class_num+a] = preds_class[train_ids[i]][a]

    # Read test data
    test_ids = list()
    with open('test.csv', 'r') as f:
        next(f)
        for line in f:
            test_ids.append(line[:-2])

    # Create the test matrix. Use the same 32 features as above
    n_test = len(test_ids)
    avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_ids)
    cluster = nx.clustering(H, nodes=test_ids)
    succs_class = {}
    preds_class = {}

    for id in test_ids:
        succ_class = np.zeros(28)
        pred_class = np.zeros(28)
        for neig in G.neighbors(id):
            if neig in train_ids:
                succ_class[y_set[y_train[train_ids.index(neig)]]] += 1
        succs_class[id] = succ_class
        for neig in G.predecessors(id):
            if neig in train_ids:
                pred_class[y_set[y_train[train_ids.index(neig)]]] += 1
        preds_class[id] = pred_class

    X_test = np.zeros((n_test, 4+class_num*2))
    for i in range(n_test):
    	X_test[i,0] = G.out_degree(test_ids[i])
    	X_test[i,1] = G.in_degree(test_ids[i])
    	X_test[i,2] = avg_neig_deg[test_ids[i]]
        X_test[i,3] = cluster[test_ids[i]]
        for a in range(class_num):
            X_test[i,4+a] = succs_class[test_ids[i]][a]
        for a in range(class_num):
            X_test[i,4+class_num+a] = preds_class[test_ids[i]][a]


    print("Train matrix dimensionality: (%d, %d)" % (X_train.shape[0], X_train.shape[1]))
    print("Test matrix dimensionality: (%d, %d)" % (X_test.shape[0], X_test.shape[1]))
    return X_train, y_train, X_test
