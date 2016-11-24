
from SampleTransformation import SampleTransformation
import numpy as np

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
            newdataset=[]
            for sample_i, sample in enumerate(dataset):
                newdataset.append(tform.apply(sample))
            dataset=np.array(newdataset)

        mean_value=np.mean(np.array(dataset))
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
            newdataset=[]
            for sample_i, sample in enumerate(dataset):
                newdataset.append(tform.apply(sample))
            dataset=np.array(newdataset)

        std_value=np.std(np.array(dataset))
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
