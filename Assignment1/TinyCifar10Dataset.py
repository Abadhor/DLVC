
import pickle as pkl
import io
import numpy as np
from ImageDataset import ImageDataset
from Cifar10Dataset import Cifar10Dataset, countLabelInstances, splitSet

class TinyCifar10Dataset(ImageDataset):
	# Subset of CIFAR10 that exposes only 10% of samples.
	
	def __init__(self, fdir, split):
		"""Ctor. fdir is a path to a directory in which the CIFAR10
		files reside (e.g. data_batch_1 and test_batch).
		split is a string that specifies which dataset split to load.
		Can be 'train' (training set), 'val' (validation set) or 'test' (test set)."""
		super(TinyCifar10Dataset, self).__init__()
		self.sample_ratio = 0.1
		
		cifar10 = Cifar10Dataset(fdir, split)
		data, labels, label_names = cifar10.getDataset()
		data_set = dict()
		data_set['data'] = data
		data_set['labels'] = labels
		
		tiny_set, remainder_set = splitSet(data_set, self.sample_ratio)
		
		self.setDataset(tiny_set['data'], tiny_set['labels'], label_names)
		