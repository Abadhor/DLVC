import io
import numpy as np

from TinyCifar10Dataset import TinyCifar10Dataset
from MiniBatchGenerator import MiniBatchGenerator
from ImageVectorizer import ImageVectorizer
import common

cifar10batchesdir=common.configs["cifar10batchesdir"]
bs=60

print ("=== Testing with TinyCifar10Dataset ===")
print ()
train = TinyCifar10Dataset(cifar10batchesdir, 'train')
minibatchgen=MiniBatchGenerator(train, 60)

print ("Dataset has "+str(train.size())+" samples")
print ("Batch generator has "+str(minibatchgen.nbatches())+" minibatches, minibatch size: "+str(bs))
print ()

data, labels, indexes=minibatchgen.batch(0)
print ("Minibatch #0 has "+str(len(data))+" samples")
print (" Data shape: "+str(data.shape))
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
data, labels, indexes=minibatchgen.batch(66)
print ("Minibatch #66 has "+str(len(data))+" samples")
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
print ()

print ("Shuffling samples")
print ()
minibatchgen.shuffle()

data, labels, indexes=minibatchgen.batch(0)
print ("Minibatch #0 has "+str(len(data))+" samples")
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
data, labels, indexes=minibatchgen.batch(66)
print ("Minibatch #66 has "+str(len(data))+" samples")
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
print ()

print ("=== Testing with ImageVectorizer ===")
print ()

train = TinyCifar10Dataset(cifar10batchesdir, 'train')
train_vectorized = ImageVectorizer(train)
minibatchgen=MiniBatchGenerator(train_vectorized, 60)

print ("Dataset has "+str(train_vectorized.size())+" samples")
print ("Batch generator has "+str(minibatchgen.nbatches())+" minibatches, minibatch size: "+str(bs))
print ()

data, labels, indexes=minibatchgen.batch(0)
print ("Minibatch #0 has "+str(len(data))+" samples")
print (" Data shape: "+str(data.shape))
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
data, labels, indexes=minibatchgen.batch(66)
print ("Minibatch #66 has "+str(len(data))+" samples")
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
print ()

print ("Shuffling samples")
print ()
minibatchgen.shuffle()

data, labels, indexes=minibatchgen.batch(0)
print ("Minibatch #0 has "+str(len(data))+" samples")
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))
data, labels, indexes=minibatchgen.batch(66)
print ("Minibatch #66 has "+str(len(data))+" samples")
print (" First 10 sample IDs: "+str(indexes[:10]).strip(']').strip('[').replace(' ',''))

#0
# Data shape: (60, 32, 32, 3)
# First 10 sample IDs: 0,1,2,3,4,5,6,7,8,9
#Minibatch #66 has 40 samples
# First 10 sample IDs: 3960,3961,3962,3963,3964,3965,3966,3967,3968,3969
