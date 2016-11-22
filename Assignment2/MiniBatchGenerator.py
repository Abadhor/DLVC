import math
from random import shuffle
import numpy as np


class MiniBatchGenerator:
    # Create minibatches of a given size from a dataset.
    # Preserves the original sample order unless shuffle() is used.

    def __init__(self, dataset, bs, tform=None):
        # Constructor.
        # dataset is a ClassificationDataset to wrap.
        # bs is an integer specifying the minibatch size.
        # tform is an optional SampleTransformation.
        # If given, tform is applied to all samples returned in minibatches.
        self.dataset=dataset
        self.bs=bs
        self.tform=tform

        self.dataids=list(range(0,self.dataset.size()))
        self.setbatches()


    def setbatches(self):
        self.batchesdataids=[]
        for i in range(0, len(self.dataids), self.bs):
            self.batchesdataids.append(self.dataids[i:i+self.bs])



    def batchsize(self):
        # Return the number of samples per minibatch.
        # The size of the last batch might be smaller.
        return [len(x) for x in self.batchesdataids]


    def nbatches(self):
        # Return the number of minibatches.
        #return math.ceil(float(len(self.data)/float(self.bs)))
        return len(self.batchesdataids)

    def shuffle(self):
        # Shuffle the dataset samples so that each
        # ends up at a random location in a random minibatch.
        shuffle(self.dataids)
        self.setbatches()

    def batch(self, bid):
        # Return the bid-th minibatch.
        # Batch IDs start with 0 and are consecutive.
        # Throws an error if the minibatch does not exist.
        indexes=self.batchesdataids[bid]
        data=[]
        labels=[]

        for index in indexes:
            dataitem, label=self.dataset.sample(index)
            if self.tform!=None:
                dataitem=self.tform.apply(dataitem)

            data.append(dataitem)
            labels.append(label)

        return np.array(data), labels, indexes
