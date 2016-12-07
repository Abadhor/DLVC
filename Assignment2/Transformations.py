
from SampleTransformation import SampleTransformation
import numpy as np
import pdb

class IdentityTransformation(SampleTransformation):
    # A transformation that does not do anything.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        return sample

class FloatCastTransformation(SampleTransformation):
    # Casts the sample datatype to single-precision float (e.g. numpy.float32).

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        return np.float32(sample)


class SubtractionTransformation(SampleTransformation):
    # Subtract a scalar from all features.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.


        if tform!=None:
            data, labels, label_names = dataset.getDataset()
            newdata=[]
            for sample in data:
                newdata.append(tform.apply(sample))
            dataset.setDataset(np.array(newdata), labels, label_names)

        data, labels, label_names = dataset.getDataset()
        mean_value=np.mean(np.array(data))
        trans=SubtractionTransformation(mean_value)

        return trans

    def __init__(self, value):
        # Constructor.
        # value is a scalar to subtract.
        self.value=value

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.
        return  np.array(sample) - self.value

    def value(self):
        # Return the subtracted value.
        return self.value

class DivisionTransformation(SampleTransformation):
    # Divide all features by a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.


        if tform!=None:
            data, labels, label_names = dataset.getDataset()
            newdata=[]
            for sample in data:
                newdata.append(tform.apply(sample))
            dataset.setDataset(np.array(newdata), labels, label_names)

        data, labels, label_names = dataset.getDataset()
        std_value=np.std(np.array(data))
        trans=DivisionTransformation(std_value)

        return trans

    def __init__(self, value):
        # Constructor.
        # value is a scalar divisor != 0.
        self.value=value

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.
        return  np.array(sample) / self.value

    def value(self):
        # Return the divisor.
        return self.value

class PerChannelSubtractionImageTransformation(SampleTransformation):
    # Perform per-channel subtraction of of image samples with a scalar.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.

        if tform!=None:
            data, labels, label_names = dataset.getDataset()
            newdata=[]
            for sample in data:
                newdata.append(tform.apply(sample))
            dataset.setDataset(np.array(newdata), labels, label_names)

        samples_data=[]
        for i in range(dataset.size()):
            samples_data.append(dataset.sample(i)[0])
        samples_data=np.array(samples_data)

        mean_values=np.mean(samples_data, axis=(0,1,2))
        trans=PerChannelSubtractionImageTransformation(mean_values)

        return trans

    def __init__(self, values):
        # Constructor.
        # values is a vector of c values to subtract, one per channel.
        # c can be any value > 0.
        self.values=values

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return  np.array(sample) - self.values

    def values(self):
        # Return the subtracted values.
        return self.values

class PerChannelDivisionImageTransformation(SampleTransformation):
    # Perform per-channel division of of image samples with a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.

        if tform!=None:
            data, labels, label_names = dataset.getDataset()
            newdata=[]
            for sample in data:
                newdata.append(tform.apply(sample))
            dataset.setDataset(np.array(newdata), labels, label_names)

        samples_data=[]
        for i in range(dataset.size()):
            samples_data.append(dataset.sample(i)[0])
        samples_data=np.array(samples_data)

        std_values=np.std(samples_data, axis=(0,1,2))
        trans=PerChannelDivisionImageTransformation(std_values)

        return trans

    def __init__(self, values):
        # Constructor.
        # values is a vector of c divisors, one per channel.
        # c can be any value > 0.
        self.values=values

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return  np.array(sample) / self.values

    def values(self):
        # Return the divisors.
        return self.values