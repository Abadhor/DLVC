from SampleTransformation import SampleTransformation

class TransformationSequence(SampleTransformation):
    # Applies a sequence of transformations
    # in the order they were added via add_transformation().

    def __init__(self):
        # Constructor.
        self.transformations=[]

    def add_transformation(self, transformation):
        # Add a transformation (type SampleTransformation) to the sequence.
        self.transformations.append(transformation)

    def get_transformation(self, tid):
        # Return the id-th transformation added via add_transformation.
        # The first transformation added has ID 0.
        return self.transformations[tid]

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        for trans in self.transformations:
            sample=trans.apply(sample)
        return sample