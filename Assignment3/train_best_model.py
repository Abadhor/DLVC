import io
import numpy as np
import tensorflow as tf

from Cifar10Dataset import Cifar10Dataset
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
from Transformations import PerChannelSubtractionImageTransformation, FloatCastTransformation, PerChannelDivisionImageTransformation
from TransformationSequence import TransformationSequence
import common

EPOCHS = 100
MOMENTUM = 0.9
LEARNING_RATE = 0.001
MINI_BATCH_SIZE = 64
SAVE_PATH = "best_model.h5"
EARLY_STOPP_EPOCH_LIMIT = 10
WEIGHT_DECAY = 0.0001
MIRROR_PROB = 0.5
CROP_WIDTH = 24
CROP_HEIGHT = 24


cifar10batchesdir=common.configs["cifar10batchesdir"]

print("Loading Cifar10Dataset ...")

train= Cifar10Dataset(cifar10batchesdir, 'train')
train_vectorized = ImageVectorizer(train)
train_data, train_labels, train_label_names=train_vectorized.getDataset()
train_labels_one_hot=np.eye(10)[train_labels]

val= Cifar10Dataset(cifar10batchesdir, 'val')
val_vectorized = ImageVectorizer(val)
val_data, val_labels, val_label_names=val_vectorized.getDataset()
val_labels_one_hot=np.eye(10)[val_labels]

test= Cifar10Dataset(cifar10batchesdir, 'test')
test_vectorized = ImageVectorizer(test)
test_data, test_labels, test_label_names=test_vectorized.getDataset()
test_labels_one_hot=np.eye(10)[test_labels]

nclasses=train.nclasses()
vectorsize=len(train_data[0])

#preprocessing
print("Setting up preprocessing...")
transformation_seq=TransformationSequence()

print(" Adding FloatCastTransformation")
floatcast_trans=FloatCastTransformation()
transformation_seq.add_transformation(floatcast_trans)

perchannelsubtraction_trans=PerChannelSubtractionImageTransformation.from_dataset_mean(train)
transformation_seq.add_transformation(perchannelsubtraction_trans)
print (" Adding PerChannelSubtractionImageTransformation [train] (value: "+str(perchannelsubtraction_trans.values)+")")

perchanneldevision_trans=PerChannelDivisionImageTransformation.from_dataset_stddev(train)
transformation_seq.add_transformation(perchanneldevision_trans)
print (" Adding PerChannelDivisionImageTransformation [train] (value: "+str(perchanneldevision_trans.values)+")")


newdata=[]
for i in range(train.size()):
    newsample=transformation_seq.apply(train.sample(i)[0])
    newdata.append(newsample)
train_data=np.array(newdata)
train.setDataset(train_data, train_labels, train_label_names)

newdata=[]
for i in range(val.size()):
    newsample=transformation_seq.apply(val.sample(i)[0])
    newdata.append(newsample)
val_data=np.array(newdata)
val.setDataset(val_data, val_labels, val_label_names)

newdata=[]
for i in range(test.size()):
    newsample=transformation_seq.apply(test.sample(i)[0])
    newdata.append(newsample)
test_data=np.array(newdata)
test.setDataset(test_data, test_labels, test_label_names)

#initializing minibatch
print("Initializing minibatch generators ...")

print("Setting up batch transformations for Training Set ...")
train_batch_tform_seq = TransformationSequence()

print(" Adding Mirror Transformation")
mirror_trans = HorizontalMirroringTransformation(MIRROR_PROB)
train_batch_tform_seq.add_transformation(mirror_trans)

print(" Adding Random Crop Transformation")
crop_trans = RandomCropTransformation(CROP_WIDTH, CROP_HEIGHT)
train_batch_tform_seq.add_transformation(crop_trans)

train_minibatchgen=MiniBatchGenerator(train, MINI_BATCH_SIZE, train_batch_tform_seq)
print(" [train] "+str(train.size())+" samples, "+str(train_minibatchgen.nbatches())+" minibatches of size "+str(train_minibatchgen.getbs())+"")


print("Setting up batch transformations for Validation Set ...")
val_batch_tform_seq = TransformationSequence()

print(" Adding Mirror Transformation")
mirror_trans = HorizontalMirroringTransformation(MIRROR_PROB)
val_batch_tform_seq.add_transformation(mirror_trans)

print(" Adding Center Crop Transformation")
center_crop_trans = CenterCropTransformation(CROP_WIDTH, CROP_HEIGHT)
val_batch_tform_seq.add_transformation(center_crop_trans)

val_minibatchgen=MiniBatchGenerator(val, 100, val_batch_tform_seq)
print(" [val] "+str(val.size())+" samples, "+str(val_minibatchgen.nbatches())+" minibatches of size "+str(val_minibatchgen.getbs())+"")

#defining NN structure
print("Initializing CNN and optimizer ...")


def variable_with_weight_decay(name, shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial)
    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    weight_decay = tf.mul(tf.nn.l2_loss(var), WEIGHT_DECAY)
    tf.add_to_collection('weight_decays', weight_decay)
    return var

def variable_bias(shape):
    initial=tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
    #return tf.get_variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

with tf.device(common.configs["devicename"]):
    #layer0 input
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
    #x_image = tf.reshape(x, [-1,32,32,3]) #batch, size, channel

    #layer1 convolution
    W_conv1 = variable_with_weight_decay('W_conv1', [3, 3, 3, 16]) #patchsize, channels, features
    b_conv1 = variable_bias([16])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #layer2 convolution
    W_conv2 = variable_with_weight_decay('W_conv2', [3, 3, 16, 32]) #patchsize, channels, features
    b_conv2 = variable_bias([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #layer3 convolution
    W_conv3 = variable_with_weight_decay('W_conv3', [3, 3, 32, 32]) #patchsize, channels, features
    b_conv3 = variable_bias([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #layer3 flatten
    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*32])
    W_fc1 = variable_with_weight_decay('W_fc1', [4*4*32, nclasses])
    b_fc1 = variable_bias([nclasses])

    y = tf.add(tf.matmul(h_pool3_flat, W_fc1), b_fc1, name="y")

    y_ = tf.placeholder(tf.float32, [None, nclasses])


total_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_) + tf.add_n(tf.get_collection('weight_decays')))

train_step = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM).minimize(total_loss)


#session
config = tf.ConfigProto()#log_device_placement=True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess=tf.InteractiveSession(config=config)

#sess = tf.InteractiveSession()

tf.initialize_all_variables().run()
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
best_model_accuracy = -1.0
best_model_epoch = -1
no_improvement_count = 0

print("Training for "+str(EPOCHS)+" epochs ...")
for epoch in range(0, EPOCHS):
    train_accuracies=[]
    train_losses=[]
    val_accuracies=[]
    for i in range(0, train_minibatchgen.nbatches()):
        batch_data, batch_labels, _ = train_minibatchgen.batch(i)
        batch_labels_one_hot=np.eye(nclasses)[batch_labels]

        sess.run(train_step, feed_dict={x: batch_data, y_: batch_labels_one_hot})

        train_accuracy, train_loss=sess.run([accuracy, total_loss], feed_dict={x: batch_data, y_: batch_labels_one_hot})

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


print("Testing best model on test set ...")
test_minibatchgen=MiniBatchGenerator(test, 100)
print(" [test] "+str(test.size())+" samples, "+str(test_minibatchgen.nbatches())+" minibatches of size "+str(test_minibatchgen.getbs())+"")

test_accuracies = []
for i in range(0, test_minibatchgen.nbatches()):
    batch_data, batch_labels, _ = test_minibatchgen.batch(i)
    batch_labels_one_hot=np.eye(10)[batch_labels]

    test_accuracy=sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels_one_hot})

    test_accuracies.append(test_accuracy)

test_accuracy = np.mean(test_accuracies)
print("Accuracy: %.2f%%" % (test_accuracy * 100))
