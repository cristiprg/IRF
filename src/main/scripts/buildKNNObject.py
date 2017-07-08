from sklearn.neighbors import NearestNeighbors

import pandas as pd
import pickle
import sys

# Input and output files as arguments
csvFilePath = sys.argv[1]
kNNObjPklFile = sys.argv[2]

# Read CSV
df = pd.read_csv(csvFilePath, usecols=[0, 1, 2], delimiter=' ', header=None)
numpyInputData = df.as_matrix()

# Build sklearn KNN object
knnobj = NearestNeighbors(n_neighbors=1000, n_jobs=-1, algorithm='kd_tree', leaf_size=100).fit(numpyInputData)

with open(kNNObjPklFile, 'wb') as output:
	pickle.dump(knnobj, output, pickle.HIGHEST_PROTOCOL)
