

import io
import numpy as np

from TinyCifar10Dataset import TinyCifar10Dataset
from Cifar10Dataset import Cifar10Dataset
from Transformations import *
from TransformationSequence import TransformationSequence
import common

cifar10batchesdir=common.configs["cifar10batchesdir"]


train = TinyCifar10Dataset(cifar10batchesdir, 'train')
#train = Cifar10Dataset(dir, 'train')

print ("Computing SubtractionTransformation from TinyCifar10Dataset [train] mean")
subtraction_trans=SubtractionTransformation.from_dataset_mean(train)
print (" Value: "+('%.2f' % subtraction_trans.value))

print ("Computing DivisionTransformation from TinyCifar10Dataset [train] stddev")
devision_trans=DivisionTransformation.from_dataset_stddev(train)
print (" Value: "+('%.2f' % devision_trans.value))

sample, label =train.sample(0)
#sample_vec, label =train_vectorized.sample(0)
print ("First sample of TinyCifar10Dataset [train]: shape: "+str(sample.shape)+", data type: "+str(sample.dtype)+", mean: "+('%.1f' % np.mean(sample))+", min: "+('%.1f' % np.min(sample))+", max: "+('%.1f' % np.max(sample)))

identity_trans=IdentityTransformation()
newsample=identity_trans.apply(sample)
print ("After applying IdentityTransformation: shape: "+str(newsample.shape)+", data type: "+str(newsample.dtype)+", mean: "+('%.1f' % np.mean(newsample))+", min: "+('%.1f' % np.min(newsample))+", max: "+('%.1f' % np.max(newsample)))

floatcast_trans=FloatCastTransformation()
newsample=floatcast_trans.apply(sample)
print ("After applying FloatCastTransformation: shape: "+str(newsample.shape)+", data type: "+str(newsample.dtype)+", mean: "+('%.1f' % np.mean(newsample))+", min: "+('%.1f' % np.min(newsample))+", max: "+('%.1f' % np.max(newsample)))


transformation_seq=TransformationSequence()
transformation_seq.add_transformation(floatcast_trans)
transformation_seq.add_transformation(subtraction_trans)
newsample=transformation_seq.apply(sample)
print ("After applying sequence FloatCast -> SubtractionTransformation: shape: "+str(newsample.shape)+", data type: "+str(newsample.dtype)+", mean: "+('%.1f' % np.mean(newsample))+", min: "+('%.1f' % np.min(newsample))+", max: "+('%.1f' % np.max(newsample)))

transformation_seq.add_transformation(devision_trans)
newsample=transformation_seq.apply(sample)
print ("After applying sequence FloatCast -> SubtractionTransformation -> DivisionTransformation: shape: "+str(newsample.shape)+", data type: "+str(newsample.dtype)+", mean: "+('%.1f' % np.mean(newsample))+", min: "+('%.1f' % np.min(newsample))+", max: "+('%.1f' % np.max(newsample)))


print ("Computing PerChannelSubtractionImageTransformation from TinyCifar10Dataset [train] mean")
perchannelsubtraction_trans=PerChannelSubtractionImageTransformation.from_dataset_mean(train)
printtext=" Values: "
for x in perchannelsubtraction_trans.values:
    printtext+=('%.2f' % x)+" "
print (printtext)

print ("Computing PerChannelDivisionImageTransformation from TinyCifar10Dataset [train] mean")
perchanneldevision_trans=PerChannelDivisionImageTransformation.from_dataset_stddev(train)
printtext=" Values: "
for x in perchanneldevision_trans.values:
    printtext+=('%.2f' % x)+" "
print (printtext)

transformation_seq=TransformationSequence()
transformation_seq.add_transformation(floatcast_trans)
transformation_seq.add_transformation(perchannelsubtraction_trans)
transformation_seq.add_transformation(perchanneldevision_trans)
newsample=transformation_seq.apply(sample)
print ("After applying sequence FloatCast -> PerChannelSubtractionImageTransformation -> PerChannelDivisionImageTransformation: shape: "+str(newsample.shape)+", data type: "+str(newsample.dtype)+", mean: "+('%.1f' % np.mean(newsample))+", min: "+('%.1f' % np.min(newsample))+", max: "+('%.1f' % np.max(newsample)))
