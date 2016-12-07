import io
import numpy as np

from TinyCifar10Dataset import TinyCifar10Dataset
from HDF5FeatureVectorDataset import HDF5FeatureVectorDataset
from MiniBatchGenerator import MiniBatchGenerator
from Transformations import SubtractionTransformation, FloatCastTransformation, DivisionTransformation
from TransformationSequence import TransformationSequence
from SoftmaxNN import SoftmaxNN

EPOCHS = 200
MOMENTUM = 0.9
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 64
SAVE_PATH = "model_best_hogoptsoftmax.h5"
EARLY_STOPP_EPOCH_LIMIT = 50

LEARNING_RATE_RANGE = [0.5, 0.1, 0.01, 0.001, 0.0001]
WEIGHT_DECAY_RANGE = [0.5, 0.25, 0.125, 0.0625, 0.03125]
#tf.logging.set_verbosity(tf.logging.ERROR)



dir = '../Data/cifar-10-batches-py'
train_file = '/features_tinycifar10_train.h5'
val_file = '/features_tinycifar10_val.h5'
test_file = '/features_tinycifar10_test.h5'


test = TinyCifar10Dataset(dir, 'test')
_, _, label_names = test.getDataset()

train_hog = HDF5FeatureVectorDataset(dir + train_file, label_names)
val_hog = HDF5FeatureVectorDataset(dir + val_file, label_names)
test_hog = HDF5FeatureVectorDataset(dir + test_file, label_names)

train_data, train_labels, train_label_names=train_hog.getDataset()
train_labels_one_hot=np.eye(10)[train_labels]

val_data, val_labels, val_label_names=val_hog.getDataset()
val_labels_one_hot=np.eye(10)[val_labels]

test_data, test_labels, test_label_names=test_hog.getDataset()
test_labels_one_hot=np.eye(10)[test_labels]

nclasses=train_hog.nclasses()
vectorsize=len(train_data[0])

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


newdataset=[]
for sample_i, sample in enumerate(test_data):
    newsample=transformation_seq.apply(sample)
    newdataset.append(newsample)
test_data=np.array(newdataset)
test_hog.setDataset(test_data, test_labels, test_label_names)

#initializing minibatch
train_minibatchgen=MiniBatchGenerator(train_hog, MINI_BATCH_SIZE)
print("Initializing minibatch generators ...")
print(" [train] "+str(train_hog.size())+" samples, "+str(train_minibatchgen.nbatches())+" minibatches of size "+str(train_minibatchgen.getbs())+"")

val_minibatchgen=MiniBatchGenerator(val_hog, 100)
print(" [val] "+str(val_hog.size())+" samples, "+str(val_minibatchgen.nbatches())+" minibatches of size "+str(val_minibatchgen.getbs())+"")


best_model_accuracy = -1.0
best_network = None
best_learning_rate = None
best_weight_decay = None

for learning_rate in LEARNING_RATE_RANGE:
    for weight_decay in WEIGHT_DECAY_RANGE:
        network = SoftmaxNN(learning_rate, MOMENTUM, weight_decay, nclasses, vectorsize)
        accuracy = network.train(train_minibatchgen, val_minibatchgen, EPOCHS, EARLY_STOPP_EPOCH_LIMIT)
        if accuracy > best_model_accuracy:
            print("New best validation accuracy, saving model to \"%s\"" % SAVE_PATH)
            best_network = network
            best_network.save(SAVE_PATH)
            best_model_accuracy = accuracy
            best_learning_rate = learning_rate
            best_weight_decay = weight_decay
            

print("Testing best model (learning rate=%.4f, weight decay=%.4f) on test set ..." % (best_learning_rate, best_weight_decay))
test_minibatchgen=MiniBatchGenerator(test_hog, 100)
print(" [test] "+str(test_hog.size())+" samples, "+str(test_minibatchgen.nbatches())+" minibatches of size "+str(test_minibatchgen.getbs())+"")
test_accuracy = best_network.test(test_minibatchgen)
print("Accuracy: %.2f%%" % (test_accuracy * 100))



