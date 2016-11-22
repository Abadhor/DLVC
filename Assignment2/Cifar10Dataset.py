
import pickle as pkl
import io
import numpy as np
from ImageDataset import ImageDataset

class Cifar10Dataset(ImageDataset):
	# The CIFAR10 dataset.
	
	def __init__(self, fdir, split):
		"""Ctor. fdir is a path to a directory in which the CIFAR10
		files reside (e.g. data_batch_1 and test_batch).
		split is a string that specifies which dataset split to load.
		Can be 'train' (training set), 'val' (validation set) or 'test' (test set)."""
		super(Cifar10Dataset, self).__init__()
		self.training_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
		self.test_batches = ['test_batch']
		self.meta = 'batches.meta'
		self.evalSet_size = 0.2
		self.trainingSet_size = 1 - self.evalSet_size
		
		if split == 'train':
			self.data, self.labels, self.label_names = self.createTrainingSet(fdir)
		elif split == 'val':
			self.data, self.labels, self.label_names = self.createValidationSet(fdir)
		elif split == 'test':
			self.data, self.labels, self.label_names = self.createTestSet(fdir)
		else:
			print("split must be 'train' (training set), 'val' (validation set) or 'test' (test set).")
		self.setDataset(self.data, self.labels, self.label_names)
	
	def createTrainingSet(self, fdir):
		merged_dict = self.mergeBatches(fdir, self.training_batches)
		training_dict, eval_dict = splitSet(merged_dict, self.trainingSet_size)
		meta_dict = self.unpickle(fdir + '/' + self.meta)
		return training_dict['data'], training_dict['labels'], meta_dict['label_names']
	
	def createValidationSet(self, fdir):
		merged_dict = self.mergeBatches(fdir, self.training_batches)
		training_dict, eval_dict = splitSet(merged_dict, self.trainingSet_size)
		meta_dict = self.unpickle(fdir + '/' + self.meta)
		return eval_dict['data'], eval_dict['labels'], meta_dict['label_names']
	
	def createTestSet(self, fdir):
		merged_dict = self.mergeBatches(fdir, self.test_batches)
		meta_dict = self.unpickle(fdir + '/' + self.meta)
		return merged_dict['data'], merged_dict['labels'], meta_dict['label_names']
	
	def unpickle(self, file):
		fo = io.open(file, 'rb')
		data_dict = pkl.load(fo, encoding='latin1')
		fo.close()
		return data_dict
	
	def mergeBatches(self, dir, batches):
		retDict = None
		for batch in batches:
			data_dict = self.unpickle(dir + '/' + batch)
			if retDict == None:
				retDict = data_dict
			else:
				first = retDict['data']
				next = data_dict['data']
				retDict['data'] = np.concatenate((first,next))
				retDict['labels'].extend(data_dict['labels'])
		return retDict
	
def countLabelInstances(labels):
	retDict = dict()
	for label in labels:
		if label in retDict:
			retDict[label] += 1
		else:
			retDict[label] = 1
	return retDict

def splitSet(data_dict, ratio):
	"""Split a set into two sets, whereas each class in set A contains 
	the first <ratio> elements of the original set and set B contains
	the last 1-<ratio> elements of the original set."""
	A_dict = dict()
	A_data = []
	A_labels = []
	B_dict = dict()
	B_data = []
	B_labels = []
	
	data = data_dict['data']
	labels = data_dict['labels']
	labelTotalCounts = countLabelInstances(labels)
	labelCurCounts = dict()
	rowIndex = 0
	for row in data:
		label = labels[rowIndex]
		if label in labelCurCounts:
			labelCurCounts[label] += 1
		else:
			labelCurCounts[label] = 1
		if labelCurCounts[label] / labelTotalCounts[label] <= ratio:
			A_data.append(row)
			A_labels.append(label)
		else:
			B_data.append(row)
			B_labels.append(label)
		
		rowIndex += 1

	A_data = np.asarray(A_data)
	A_dict['data'] = A_data
	A_dict['labels'] = A_labels
	
	B_data = np.asarray(B_data)
	B_dict['data'] = B_data
	B_dict['labels'] = B_labels
	
	#print(A_data.shape) #should be 40000
	#print(B_data.shape) #should be 10000
	return A_dict, B_dict