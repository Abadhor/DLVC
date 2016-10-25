


import pickle as pkl
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer



dir = '../Data/cifar-10-batches-py'


train = TinyCifar10Dataset(dir, 'train')

train_vectorized = ImageVectorizer(train)
class_num = 1
sample_num = 499

print(str(train_vectorized.size())+" samples")
print(str(train_vectorized.nclasses())+" classes, name of class #"+str(class_num)+": "+train_vectorized.classname(class_num))

vec, label = train_vectorized.sample(sample_num)
print("Sample #"+str(sample_num)+": "+train_vectorized.classname(label)+", shape: "+str(vec.shape))
print("Shape after devectorization: "+str(train_vectorized.devectorize(vec).shape))

im = train_vectorized.devectorize(vec)
plt.imsave("00_vectorized_horse.png", im)