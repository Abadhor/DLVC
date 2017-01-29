import io
import numpy as np
import tensorflow as tf

from Cifar10Dataset import Cifar10Dataset
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
from Transformations import *
from TransformationSequence import TransformationSequence
import common

EPOCHS = 200
MOMENTUM = 0.9
INIT_LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 64
SAVE_PATH = "/dlvc/assignments/assignment3/group4/best_model.h5"
EARLY_STOPP_EPOCH_LIMIT = 20
LEARN_RATE_DECAY_EPOCH_LIMIT = 5
LR_DECAY_RATE = 0.1
WEIGHT_DECAY = 0.0001
MIRROR_PROB = 0.5
CROP_WIDTH = 24
CROP_HEIGHT = 24


cifar10batchesdir=common.configs["cifar10batchesdir"]

print("Loading Cifar10Dataset ...")

train= Cifar10Dataset(cifar10batchesdir, 'train')


val= Cifar10Dataset(cifar10batchesdir, 'val')


nclasses=train.nclasses()

#initializing minibatch
print("Initializing minibatch generators ...")

print("Setting up batch transformations for Training Set ...")
train_batch_tform_seq = TransformationSequence()

print(" Adding FloatCastTransformation")
floatcast_trans=FloatCastTransformation()
train_batch_tform_seq.add_transformation(floatcast_trans)

perchannelsubtraction_trans=PerChannelSubtractionImageTransformation.from_dataset_mean(train)
train_batch_tform_seq.add_transformation(perchannelsubtraction_trans)
print (" Adding PerChannelSubtractionImageTransformation [train] (value: "+str(perchannelsubtraction_trans.values)+")")

perchanneldevision_trans=PerChannelDivisionImageTransformation.from_dataset_stddev(train)
train_batch_tform_seq.add_transformation(perchanneldevision_trans)
print (" Adding PerChannelDivisionImageTransformation [train] (value: "+str(perchanneldevision_trans.values)+")")

print(" Adding Mirror Transformation")
mirror_trans = HorizontalMirroringTransformation(MIRROR_PROB)
train_batch_tform_seq.add_transformation(mirror_trans)

print(" Adding Random Crop Transformation")
crop_trans = RandomCropTransformation(CROP_WIDTH, CROP_HEIGHT)
train_batch_tform_seq.add_transformation(crop_trans)

train_minibatchgen=MiniBatchGenerator(train, MINI_BATCH_SIZE, train_batch_tform_seq)
print(" [train] "+str(train.size())+" samples, "+str(train_minibatchgen.nbatches())+" minibatches of size "+str(train_minibatchgen.getbs())+"")


val_minibatchgen=MiniBatchGenerator(val, 100, train_batch_tform_seq)
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

def variable_bias(name, shape):
    #initial=tf.constant(0.1, shape=shape)
    #return tf.Variable(initial)
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_strides(x, W, strides):
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def res(x, in_channels, out_channels, layer_name):
    #layer1 convolution
    W_conv1 = variable_with_weight_decay(layer_name+'_W_conv1', [3, 3, in_channels, out_channels]) #patchsize, channels, features
    b_conv1 = variable_bias(layer_name+'_b_conv1',[out_channels])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    #layer2 convolution
    W_conv2 = variable_with_weight_decay(layer_name+'_W_conv2', [3, 3, in_channels, out_channels]) #patchsize, channels, features
    b_conv2 = variable_bias(layer_name+'_b_conv2',[out_channels])
    h_conv2 = tf.nn.relu((conv2d(h_conv1, W_conv2) + b_conv2) + x)
    return h_conv2


# #basic architecture
# with tf.device(common.configs["devicename"]):
    # #layer0 input
    # x = tf.placeholder(tf.float32, [None, 24, 24, 3], name="x")
    # #x_image = tf.reshape(x, [-1,32,32,3]) #batch, size, channel

    # #layer1 convolution
    # W_conv1 = variable_with_weight_decay('W_conv1', [3, 3, 3, 16]) #patchsize, channels, features
    # b_conv1 = variable_bias([16])
    # h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)

    # #layer2 convolution
    # W_conv2 = variable_with_weight_decay('W_conv2', [3, 3, 16, 32]) #patchsize, channels, features
    # b_conv2 = variable_bias([32])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    # #layer3 convolution
    # W_conv3 = variable_with_weight_decay('W_conv3', [3, 3, 32, 32]) #patchsize, channels, features
    # b_conv3 = variable_bias([32])
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # #layer3 flatten
    # h_pool3_flat = tf.reshape(h_pool3, [-1, 3*3*32])
    # W_fc1 = variable_with_weight_decay('W_fc1', [3*3*32, 128])
    # b_fc1 = variable_bias([128])
    # #W_fc1 = variable_with_weight_decay('W_fc1', [3*3*32, nclasses])
    # #b_fc1 = variable_bias([nclasses])


    # h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    # #dropout
    # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # #second fully connected layer
    # W_fc2 = variable_with_weight_decay('W_fc2', [128, nclasses])
    # b_fc2 = variable_bias([nclasses])
    
    # y = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="y")

    # y_ = tf.placeholder(tf.float32, [None, nclasses])

#res net architecture
with tf.device(common.configs["devicename"]):
    #layer0 input
    x = tf.placeholder(tf.float32, [None, 24, 24, 3], name="x")
    #x_image = tf.reshape(x, [-1,32,32,3]) #batch, size, channel
    
    W_conv01 = variable_with_weight_decay('W_conv01', [3, 3, 3, 32]) #patchsize, channels, features
    b_conv01 = variable_bias('b_conv01',[32])
    h_conv01 = tf.nn.relu(conv2d_strides(x, W_conv01, 1) + b_conv01)
    
    #res01
    res01 = res(h_conv01, 32, 32, 'res01')
    
    #res02
    res02 = res(res01, 32, 32, 'res02')
    
    #res03
    res03 = res(res02, 32, 32, 'res03')

    #layer1 convolution
    #[24,24,3] -> [12,12,64]
    W_conv1 = variable_with_weight_decay('W_conv1', [5, 5, 32, 64]) #patchsize, channels, features
    b_conv1 = variable_bias('b_conv1',[64])
    h_conv1 = tf.nn.relu(conv2d_strides(res03, W_conv1, 2) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)
    
    #res1
    res1 = res(h_conv1, 64, 64, 'res1')
    
    #res2
    res2 = res(res1, 64, 64, 'res2')
    
    #res3
    res3 = res(res2, 64, 64, 'res3')
    
    #res11
    #res11 = res(res3, 64, 64, 'res11')
    
    #res21
    #res21 = res(res11, 64, 64, 'res21')
    
    #res31
    #res31 = res(res21, 64, 64, 'res31')

    #layer2 convolution
    #[12,12,64] -> [6,6,128]
    W_conv2 = variable_with_weight_decay('W_conv2', [3, 3, 64, 128]) #patchsize, channels, features
    b_conv2 = variable_bias('b_conv2',[128])
    h_conv2 = tf.nn.relu(conv2d_strides(res3, W_conv2, 2) + b_conv2)
    
    #res4
    res4 = res(h_conv2, 128, 128, 'res4')
    
    #res5
    res5 = res(res4, 128, 128, 'res5')
    
    #res6
    res6 = res(res5, 128, 128, 'res6')
    
    #res41
    #res41 = res(res6, 128, 128, 'res41')
    
    #res51
    #res51 = res(res41, 128, 128, 'res51')
    
    #res61
    #res61 = res(res51, 128, 128, 'res61')

    #layer3 convolution
    #[6,6,128] -> [3,3, 256]
    W_conv3 = variable_with_weight_decay('W_conv3', [3, 3, 128, 256]) #patchsize, channels, features
    b_conv3 = variable_bias('b_conv3',[256])
    h_conv3 = tf.nn.relu(conv2d_strides(res6, W_conv3, 2) + b_conv3)

    #layer3 flatten
    h_pool3_flat = tf.reshape(h_conv3, [-1, 3*3*256])
    W_fc1 = variable_with_weight_decay('W_fc1', [3*3*256, 1024])
    b_fc1 = variable_bias('b_fc1',[1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    #dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #second fully connected layer
    W_fc2 = variable_with_weight_decay('W_fc2', [1024, nclasses])
    b_fc2 = variable_bias('b_fc2',[nclasses])
    
    y = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="y")

    y_ = tf.placeholder(tf.float32, [None, nclasses])


total_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_) + tf.add_n(tf.get_collection('weight_decays')))

learning_rate = tf.Variable(INIT_LEARNING_RATE, name='learning_rate')

train_step = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(total_loss)


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
    train_minibatchgen.shuffle()
    for i in range(0, train_minibatchgen.nbatches()):
        batch_data, batch_labels, _ = train_minibatchgen.batch(i)
        batch_labels_one_hot=np.eye(nclasses)[batch_labels]

        sess.run(train_step, feed_dict={x: batch_data, y_: batch_labels_one_hot, keep_prob: 0.5})

        train_accuracy, train_loss=sess.run([accuracy, total_loss], feed_dict={x: batch_data, y_: batch_labels_one_hot, keep_prob: 0.5})

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        #break

    for i in range(0, val_minibatchgen.nbatches()):
        batch_data, batch_labels, _ = val_minibatchgen.batch(i)
        batch_labels_one_hot=np.eye(nclasses)[batch_labels]

        val_accuracy=sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels_one_hot, keep_prob: 1.0})

        val_accuracies.append(val_accuracy)

    epoch_validation_accuracy = np.mean(val_accuracies)

    print ("[Epoch "+('%3d' % epoch)+"] loss: "+('%.3f' % np.mean(train_losses))+", training accuracy: "+('%.3f' % np.mean(train_accuracies))+", validation accuracy: "+('%.3f' % epoch_validation_accuracy))
    
    if epoch_validation_accuracy > best_model_accuracy:
        print("New best validation accuracy, saving model to \"%s\"" % SAVE_PATH)
        best_model_accuracy = epoch_validation_accuracy
        best_model_epoch = epoch
        save_path = saver.save(sess, SAVE_PATH)
        no_improvement_count = 0
        no_improvement_count_lr = 0
    else:
        no_improvement_count += 1
        no_improvement_count_lr += 1

    if no_improvement_count >= EARLY_STOPP_EPOCH_LIMIT:
        print("Validation accuracy did not improve for %i epochs, stopping" % EARLY_STOPP_EPOCH_LIMIT)
        break
    
    if no_improvement_count_lr >= LEARN_RATE_DECAY_EPOCH_LIMIT:
        print("Validation accuracy did not improve for %i epochs, decreasing learning rate" % LEARN_RATE_DECAY_EPOCH_LIMIT)
        learning_rate.assign(learning_rate*LR_DECAY_RATE)
        no_improvement_count_lr = 0

print("Best validation accuracy: %.3f (epoch %i)" % (best_model_accuracy, best_model_epoch))

