
from ClassificationDataset import ClassificationDataset

class ImageDataset(ClassificationDataset):
	# A dataset, consisting of multiple samples/images
	# and corresponding class labels.
	
	def sample(self, sid):
		"""Returns the sid-th sample in the dataset, and the
		corresponding class label. Depending of your language,
		this can be a Matlab struct, Python tuple or dict, etc.
		Sample IDs start with 0 and are consecutive.
		The channel order of samples must be RGB.
		Throws an error if the sample does not exist."""
		return self.data[sid, :].reshape(3,32,32).transpose(1,2,0), self.labels[sid]