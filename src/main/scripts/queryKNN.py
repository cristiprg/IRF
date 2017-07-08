from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import pickle
import sys
from StringIO import StringIO


# Input and output files as arguments
kNNObjPklFile = sys.argv[1]


def getKNNObject():
	with open(kNNObjPklFile, 'rb') as input:
		return pickle.load(input)

def getQueryData():
	df = pd.read_csv(sys.stdin, delimiter=',', header=None)
	return df.as_matrix()


if __name__ == "__main__":
	knn_object = getKNNObject()
	query_set = getQueryData()
	nearest_neighbors_ids = knn_object.kneighbors(query_set, n_neighbors=100, return_distance=False)
	
	# Correctly formatting the output is essential. One line -> one array of ids
	np.set_printoptions(threshold=np.inf)

	for array in nearest_neighbors_ids:
		sep = ''
		for el in array:
			sys.stdout.write(sep + str(el))
			sep = ','
		sys.stdout.write('\n')

