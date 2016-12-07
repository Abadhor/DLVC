import argparse
from scipy.misc import imread
import pdb
import tensorflow as tf
from Transformations import PerChannelSubtractionImageTransformation, FloatCastTransformation, \
    PerChannelDivisionImageTransformation, ResizeImageTransformation
from TransformationSequence import TransformationSequence
import numpy as np
from datetime import datetime

print ('Parsing arguments ...')
parser = argparse.ArgumentParser()

parser.add_argument('--model', action='store', dest='modelpath',
                    help='path to a classifier ')
parser.add_argument('--image', action='store', dest='imagepath',
                    help='path to a RGB input image')
parser.add_argument('--means', action='store', dest='means', nargs='+', type=float,
                    help='means for preprocessing')
parser.add_argument('--stds', action='store', dest='stds', nargs='+', type=float,
                    help='standard deviations for preprocessing')

args = parser.parse_args()

print (' Model: '+args.modelpath)
print (' Image: '+args.imagepath)
print (' Means: '+str(args.means))
print (' Stds: '+str(args.stds))


print ('Loading image ...')
im = imread(args.imagepath)
print (' Shape: '+str(im.shape))


print ('Loading classifier ...')
sess=tf.Session()
saver = tf.train.import_meta_graph(args.modelpath+'.meta')
saver.restore(sess, args.modelpath)

grph = tf.get_default_graph()
x = grph.get_tensor_by_name("x:0")
y = grph.get_tensor_by_name("y:0")
input_shape=(int(x.get_shape()[1]), int(x.get_shape()[2]), int(x.get_shape()[3]))
nclasses=int(y.get_shape()[1])
print (' Input shape: '+str(input_shape)+', '+str(nclasses)+' classes')

#preprocessing
print("Preprocessing image...")
print(" Transformations in order:")
transformation_seq=TransformationSequence()

print("  ResizeImageTransformation")
resizeimage_trans=ResizeImageTransformation(min(input_shape[:2]))
transformation_seq.add_transformation(resizeimage_trans)

print("  FloatCastTransformation")
floatcast_trans=FloatCastTransformation()
transformation_seq.add_transformation(floatcast_trans)

perchannelsubtraction_trans=PerChannelSubtractionImageTransformation(args.means)
transformation_seq.add_transformation(perchannelsubtraction_trans)
print ("  PerChannelSubtractionImageTransformation ("+str(perchannelsubtraction_trans.values)+")")

perchanneldevision_trans=PerChannelDivisionImageTransformation(args.stds)
transformation_seq.add_transformation(perchanneldevision_trans)
print ("  PerChannelDivisionImageTransformation ("+str(perchanneldevision_trans.values)+")")

sample=transformation_seq.apply(im)

print (" Result: shape: "+str(sample.shape)+", dtype: "+str(sample.dtype)+", mean: "+("%.3f" % np.mean(sample))+", std: "+("%.3f" % np.std(sample)))


print ("Classifying image ...")
check_time=datetime.now()

sample=np.reshape(sample, [-1,input_shape[0],input_shape[1],input_shape[2]])
prediction=sess.run(tf.nn.softmax(y), feed_dict={x: sample})[0]
dur=datetime.now()-check_time
print ("1/1 [==============================] - "+str(int(dur.total_seconds()))+"s")
prediction=np.round(prediction, 2)

printtext=" Class scores: ["
for p in prediction:
    printtext+=('%0.002f' % p)+" "
print(printtext.strip()+"]")
print(" ID of most likely class: "+str(np.argmax(prediction))+" (score: "+("%.2f" % np.max(prediction))+")")