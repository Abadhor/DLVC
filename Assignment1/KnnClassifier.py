

import numpy as np
from operator import itemgetter

class KnnClassifier:
	# k-nearest-neighbors classifier.
	
	def __init__(self, k, cmp):
		"""Ctor. k is the number of nearest neighbors to search for,
		and cmp is a string specifying the distance measure to
		use, namely `l1` (L1 distance) or `l2` (L2 distance)."""
		self.k = k
		self.cmp = cmp
	
	def train(self, dataset):
		"""Train on a dataset (type FeatureVectorDataset)."""
		# Knn requires no training
		self.dataset = dataset
	
	def predict(self, fvec):
		"""Return the predicted class label for a given feature vector fvec.
		If the label is ambiguous, any of those in question is returned."""
		distance_list = []
		
		for i in range(0, self.dataset.size()):
			sample, label = self.dataset.sample(i)
			distance = self.getDistance(fvec, sample)
			distance_list.append((label, distance))
		#sort on distances and count labels of k nearest items
		sorted_distance = sorted(distance_list, key=itemgetter(1))
		label_votes = dict()
		for i in range(0,self.k):
			if sorted_distance[i][0] in label_votes:
				label_votes[sorted_distance[i][0]] += 1
			else:
				label_votes[sorted_distance[i][0]] = 1
		#return label of that occured most often in range k
		max_tuple = max(label_votes.items(), key=itemgetter(1))
		return max_tuple[0]
	
	
	def getDistance(self, A, B):
		if self.cmp == 'l1':
			return np.linalg.norm(A - B, 1)
		elif self.cmp == 'l2':
			return np.linalg.norm(A - B, 2)
		return None
	