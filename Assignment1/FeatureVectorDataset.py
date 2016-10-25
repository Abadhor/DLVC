



class FeatureVectorDataset:
	# A dataset, consisting of multiple feature vectors
	# and corresponding class labels.
	
	
	def __init__(self):
		self.data = None
		self.labels = None
		self.label_names = None
	
	def setDataset(self, data, labels, label_names):
		self.data = data
		self.labels = labels
		self.label_names = label_names
	
	def getDataset(self):
		return self.data, self.labels, self.label_names
	
	def size(self):
		"""Returns the size of the dataset (number of images)."""
		return len(self.labels)
	
	def nclasses(self):
		"""Returns the number of different classes.
		Class labels start with 0 and are consecutive."""
		return len(self.label_names)
	
	def classname(self, cid):
		"""Returns the name of a class as a string."""
		return self.label_names[cid]
	
	def sample(self, sid):
		"""Returns the sid-th sample in the dataset, and the
		corresponding class label. Depending of your language,
		this can be a Matlab struct, Python tuple or dict, etc.
		Sample IDs start with 0 and are consecutive.
		The channel order of samples must be RGB.
		Throws an error if the sample does not exist."""
		return self.data[sid, :], self.labels[sid]