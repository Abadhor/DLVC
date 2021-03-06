import io
import numpy as np
import tensorflow as tf

from TinyCifar10Dataset import TinyCifar10Dataset
from HDF5FeatureVectorDataset import HDF5FeatureVectorDataset
from MiniBatchGenerator import MiniBatchGenerator
from Transformations import SubtractionTransformation, FloatCastTransformation, DivisionTransformation
from TransformationSequence import TransformationSequence
import common

EPOCHS = 200
MOMENTUM = 0.9
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 64
SAVE_PATH = "model_best_hogsoftmax.h5"
EARLY_STOPP_EPOCH_LIMIT = 50

cifar10batchesdir=common.configs["cifar10batchesdir"]
train_file = common.configs['tinycifar10hog.trainfile']
val_file = common.configs['tinycifar10hog.trainfile']
test_file = common.configs['tinycifar10hog.trainfile']


test = TinyCifar10Dataset(cifar10batchesdir, 'test')
_, _, label_names = test.getDataset()

train_hog = HDF5FeatureVectorDataset(train_file, label_names)
val_hog = HDF5FeatureVectorDataset(val_file, label_names)
test_hog = HDF5FeatureVectorDataset(test_file, label_names)

train_data, train_labels, train_label_names=train_hog.getDataset()
train_labels_one_hot=np.eye(10)[train_labels]

val_data, val_labels, val_label_names=val_hog.getDataset()
val_labels_one_hot=np.eye(10)[val_labels]

nclasses=train_hog.nclasses()
vectorsize=len(train_data[0])

print (nclasses, vectorsize)

#preprocessing
print("Setting up preprocessing ...")
print(" Adding FloatCastTransformation")
float_trans=FloatCastTransformation()
subtraction_trans=SubtractionTransformation.from_dataset_mean(train_hog, float_trans)
print (" Adding SubtractionTransformation [train] (value: "+('%.2f' % subtraction_trans.value)+")")

devision_trans=DivisionTransformation.from_dataset_stddev(train_hog, float_trans)
print (" Adding DivisionTransformation [train] (value: "+('%.2f' % devision_trans.value)+")")

#apply transformation
transformation_seq=TransformationSequence()
transformation_seq.add_transformation(float_trans)
transformation_seq.add_transformation(subtraction_trans)
transformation_seq.add_transformation(devision_trans)

newdataset=[]
for sample_i, sample in enumerate(train_data):
    newsample=transformation_seq.apply(sample)
    newdataset.append(newsample)
train_data=np.array(newdataset)
train_hog.setDataset(train_data, train_labels, train_label_names)

newdataset=[]
for sample_i, sample in enumerate(val_data):
    newsample=transformation_seq.apply(sample)
    newdataset.append(newsample)
val_data=np.array(newdataset)
val_hog.setDataset(val_data, val_labels, val_label_names)

#initializing minibatch
train_minibatchgen=MiniBatchGenerator(train_hog, MINI_BATCH_SIZE)
print("Initializing minibatch generators ...")
print(" [train] "+str(train_hog.size())+" samples, "+str(train_minibatchgen.nbatches())+" minibatches of size "+str(train_minibatchgen.getbs())+"")

val_minibatchgen=MiniBatchGenerator(val_hog, 100)
print(" [val] "+str(val_hog.size())+" samples, "+str(val_minibatchgen.nbatches())+" minibatches of size "+str(val_minibatchgen.getbs())+"")

#defining NN structure
x = tf.placeholder(tf.float32, [None, vectorsize])
W = tf.Variable(tf.zeros([vectorsize, nclasses]), name="weights")
b = tf.Variable(tf.zeros([nclasses]), name="biases")
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, nclasses])

loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
train_step = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM).minimize(loss_function)


#session
print("Initializing softmax classifier and optimizer ...")

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#saver for best model
saver = tf.train.Saver()
best_model_accuracy = -1.0
best_model_epoch = -1
no_improvement_count = 0

for epoch in range(0, EPOCHS):
    train_accuracies=[]
    train_losses=[]
    val_accuracies=[]
    for i in range(0, train_minibatchgen.nbatches()):
        batch_data, batch_labels, _ = train_minibatchgen.batch(i)
        batch_labels_one_hot=np.eye(nclasses)[batch_labels]

        sess.run(train_step, feed_dict={x: batch_data, y_: batch_labels_one_hot})

        train_accuracy, train_loss=sess.run([accuracy, loss_function], feed_dict={x: batch_data, y_: batch_labels_one_hot})

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        #break

    for i in range(0, val_minibatchgen.nbatches()):
        batch_data, batch_labels, _ = val_minibatchgen.batch(i)
        batch_labels_one_hot=np.eye(nclasses)[batch_labels]

        val_accuracy=sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels_one_hot})

        val_accuracies.append(val_accuracy)

    epoch_validation_accuracy = np.mean(val_accuracies)

    print ("[Epoch "+('%3d' % epoch)+"] loss: "+('%.3f' % np.mean(train_losses))+", training accuracy: "+('%.3f' % np.mean(train_accuracies))+", validation accuracy: "+('%.3f' % epoch_validation_accuracy))
    
    if epoch_validation_accuracy > best_model_accuracy:
        print("New best validation accuracy, saving model to \"%s\"" % SAVE_PATH)
        best_model_accuracy = epoch_validation_accuracy
        best_model_epoch = epoch
        save_path = saver.save(sess, SAVE_PATH)
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= EARLY_STOPP_EPOCH_LIMIT:
        print("Validation accuracy did not improve for %i epochs, stopping" % EARLY_STOPP_EPOCH_LIMIT)
        break

print("Best validation accuracy: %.3f (epoch %i)" % (best_model_accuracy, best_model_epoch))
