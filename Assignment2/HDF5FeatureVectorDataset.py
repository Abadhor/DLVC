
import h5py
import numpy as np
from FeatureVectorDataset import FeatureVectorDataset

class HDF5FeatureVectorDataset(FeatureVectorDataset):
	# A dataset stored in a HDF5 file including the datasets
	# features (n*f matrix) and labels (n-vector).
	
	def __init__(self, fpath, class_names):
		"""Ctor. fpath is a path to the HDF5 file.
		class_names is a mapping from labels to
		class names, for every label."""
		f = h5py.File(fpath, "r")
		data = np.asarray(f.get('features'))
		labels = np.asarray(f.get('labels'))
		self.setDataset(data, labels, class_names)
		