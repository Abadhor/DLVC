

import io
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

from TinyCifar10Dataset import TinyCifar10Dataset
from Cifar10Dataset import Cifar10Dataset
from Transformations import *
from TransformationSequence import TransformationSequence
import common

MIRROR_PROB = 0.5
CROP_WIDTH = 48
CROP_HEIGHT = 48

cifar10batchesdir=common.configs["cifar10batchesdir"]

imagepath = "cat.jpg"
imageName = "cat.jpg"
sample = imread(imagepath)
savepath = "augmented_samples/"

print("Creating Transformations")
mirror_trans = HorizontalMirroringTransformation(MIRROR_PROB)
crop_trans = RandomCropTransformation(CROP_WIDTH, CROP_HEIGHT)

transformation_seq=TransformationSequence()
transformation_seq.add_transformation(mirror_trans)
transformation_seq.add_transformation(crop_trans)

print("Applying Transformations & Saving Images")
for i in range(0,10):
    newsample=transformation_seq.apply(sample)
    imsave(savepath+("%02.i_" % i)+imageName, newsample)

