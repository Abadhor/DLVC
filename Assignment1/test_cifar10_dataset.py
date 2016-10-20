

import pickle as pkl
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ImageDataset import ImageDataset
from Cifar10Dataset import Cifar10Dataset, countLabelInstances, splitSet
from TinyCifar10Dataset import TinyCifar10Dataset

dir = '../Data/cifar-10-batches-py'


train = Cifar10Dataset(dir, 'train')
val = Cifar10Dataset(dir, 'val')
test = Cifar10Dataset(dir, 'test')

class_num = 1
print("[train] "+str(train.nclasses())+" classes, name of class #"+str(class_num)+": "+ train.classname(class_num))
print("[val] "+str(val.nclasses())+" classes, name of class #"+str(class_num)+": "+ val.classname(class_num))
print("[test] "+str(test.nclasses())+" classes, name of class #"+str(class_num)+": "+ test.classname(class_num))

print("[train] "+str(train.size())+" samples")
data, labels, label_names = train.getDataset()
labelTotalCounts = countLabelInstances(labels)
for i in range(0, train.nclasses()):
	print("Class #"+str(i)+": "+str(labelTotalCounts[i])+" samples")

print("[val] "+str(val.size())+" samples")
data, labels, label_names = val.getDataset()
labelTotalCounts = countLabelInstances(labels)
for i in range(0, val.nclasses()):
	print("Class #"+str(i)+": "+str(labelTotalCounts[i])+" samples")

print("[test] "+str(test.size())+" samples")
data, labels, label_names = test.getDataset()
labelTotalCounts = countLabelInstances(labels)
for i in range(0, test.nclasses()):
	print("Class #"+str(i)+": "+str(labelTotalCounts[i])+" samples")

sample_num = 499
train_sample_img, train_sample_label = train.sample(499)
im = train_sample_img
plt.imsave("00_normal_horse.png", im)

eval_sample_img, eval_sample_label = val.sample(499)
im = eval_sample_img
plt.imsave("00_normal_deer.png", im)

test_sample_img, test_sample_label = test.sample(499)
im = test_sample_img
plt.imsave("00_normal_airplane.png", im)


print("[train] Sample #"+str(sample_num)+": "+train.classname(train_sample_label))
print("[val] Sample #"+str(sample_num)+": "+val.classname(eval_sample_label))
print("[test] Sample #"+str(sample_num)+": "+test.classname(test_sample_label))


