
import pickle as pkl
import io
import numpy as np
from ImageVectorizer import ImageVectorizer
from TinyCifar10Dataset import TinyCifar10Dataset
from KnnClassifier import KnnClassifier
import random

#from sklearn import neighbors
random.seed(1)

def getAccuracy(classifier, test_set):
	label_correct = 0
	size = test_set.size()
	#size = 10
	for i in range(0, size):
		fvec, label = test_set.sample(i)
		pred_label = classifier.predict(fvec)
		if pred_label == label:
			label_correct += 1
		#print ("label: "+str(label)+" pred_label: "+str(pred_label))
	return label_correct / size

dir = '../Data/cifar-10-batches-py'


train = TinyCifar10Dataset(dir, 'train')
val = TinyCifar10Dataset(dir, 'val')
test = TinyCifar10Dataset(dir, 'test')

train_vectorized = ImageVectorizer(train)
val_vectorized = ImageVectorizer(val)
test_vectorized = ImageVectorizer(test)

print("Performing random hyperparameter search ...")
print("[train] "+str(train_vectorized.size())+" samples")
print("[val] "+str(val_vectorized.size())+" samples")

best_accuracy = 0
best_k = None
best_cmp = None

# search twenty, keep best
k = 1
cmp = 'l2'
cl = KnnClassifier(k, cmp)
cl.train(train_vectorized)
accuracy = getAccuracy(cl, val_vectorized)
print("k="+str(k)+", cmp="+cmp+", accuracy: "+str(accuracy*100)+"%")
if accuracy > best_accuracy:
	best_accuracy = accuracy
	best_k = k
	best_cmp = cmp

for i in range(0,20):
	k = random.randint(1,40)
	cmp = random.choice(['l1','l2'])
	cl = KnnClassifier(k, cmp)
	cl.train(train_vectorized)
	accuracy = getAccuracy(cl, val_vectorized)
	print("k="+str(k)+", cmp="+cmp+", accuracy: "+str(accuracy*100)+"%")
	if accuracy > best_accuracy:
		best_accuracy = accuracy
		best_k = k
		best_cmp = cmp

print("Testing best combination ("+str(best_k)+", "+best_cmp+") on test set ...")
k = best_k
cmp = best_cmp
cl = KnnClassifier(k, cmp)
cl.train(train_vectorized)
accuracy = getAccuracy(cl, test_vectorized)
print("[test] "+str(test_vectorized.size())+" samples")
print("Accuracy: "+str(accuracy*100)+"%")

# clf = neighbors.KNeighborsClassifier(1)
# fvecs = []
# labels = []
# for i in range(0, train_vectorized.size()):
	# fvec, label = train_vectorized.sample(i)
	# fvecs.append(fvec)
	# labels.append(label)
# fvecs = np.array(fvecs)
# labels = np.array(labels)
# clf.fit(fvecs, labels)

# label_correct = 0
# size = val_vectorized.size()
# #size = 10
# for i in range(0, size):
	# fvec, label = val_vectorized.sample(i)
	# pred_label = clf.predict(np.array(fvec).reshape(1,-1))
	# if pred_label == label:
		# label_correct += 1
	# #print ("label: "+str(label)+" pred_label: "+str(pred_label))
# print("Sklearn:")
# print( label_correct / size*100)
