
import numpy as np
from FeatureVectorDataset import FeatureVectorDataset

class ImageVectorizer(FeatureVectorDataset):
	# Wraps an image dataset and exposes its contents.
	# Samples obtained using sample() are returned as 1D feature vectors.
	# Use devectorize() to convert a vector back to an image.
	
	def __init__(self, dataset):
		"""Ctor. dataset is the dataset to wrap (type ImageDataset)."""
		self.dataset = dataset
		data, labels, label_names = dataset.getDataset()
		self.setDataset(data, labels, label_names)

		self.rows=32
		self.cols=32
		self.channels=3
	
	def devectorize(self, fvec):
		"""Convert a feature vector fvec obtained using sample()
		back to an image and return the converted version."""
		return fvec.reshape(self.channels,self.rows,self.cols).transpose(1,2,0)