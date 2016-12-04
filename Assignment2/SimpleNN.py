__author__='navid'

import tensorflow as tf
import numpy as np

class SimpleNN():

    def __init__(self, learning_rate, momentum, weight_decay, nclasses, vectorsize):

        self.nclasses=nclasses
        self.vectorsize=vectorsize
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        #defining NN structure
        self.x = tf.placeholder(tf.float32, [None, vectorsize])
        self.W = tf.Variable(tf.random_normal([vectorsize, nclasses], stddev=0.1), name="weights")
        self.b = tf.Variable(tf.zeros([nclasses]), name="biases")
        self.y = tf.matmul(self.x, self.W) + self.b
        self.y_ = tf.placeholder(tf.float32, [None, nclasses])

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_) + weight_decay * tf.nn.l2_loss(self.W))

        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        self.train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss_function)

    def train(self, train_minibatchgen, val_minibatchgen, epochs, early_stop_epoch_limit):
        #session
        #print("Initializing softmax classifier and optimizer ...")

        sess = tf.InteractiveSession()
        self.sess = sess

        tf.initialize_all_variables().run()
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #saver for best model
        self.saver = tf.train.Saver()
        best_model_accuracy = -1.0
        best_model_epoch = -1
        no_improvement_count = 0

        for epoch in range(0, epochs):
            train_accuracies=[]
            train_losses=[]
            val_accuracies=[]
            for i in range(0, train_minibatchgen.nbatches()):
                batch_data, batch_labels, _ = train_minibatchgen.batch(i)
                batch_labels_one_hot=np.eye(10)[batch_labels]

                sess.run(self.train_step, feed_dict={self.x: batch_data, self.y_: batch_labels_one_hot})


                train_accuracy, train_loss=sess.run([self.accuracy, self.loss_function], feed_dict={self.x: batch_data, self.y_: batch_labels_one_hot})

                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)

            #break

            for i in range(0, val_minibatchgen.nbatches()):
                batch_data, batch_labels, _ = val_minibatchgen.batch(i)
                batch_labels_one_hot=np.eye(10)[batch_labels]

                val_accuracy=sess.run(self.accuracy, feed_dict={self.x: batch_data, self.y_: batch_labels_one_hot})

                val_accuracies.append(val_accuracy)

            epoch_validation_accuracy = np.mean(val_accuracies)

            #print ("[Epoch "+('%3d' % epoch)+"] loss: "+('%.3f' % np.mean(train_losses))+", training accuracy: "+('%.3f' % np.mean(train_accuracies))+", validation accuracy: "+('%.3f' % epoch_validation_accuracy))

            if epoch_validation_accuracy > best_model_accuracy:
                #print("New best validation accuracy, saving model to \"%s\"" % SAVE_PATH)
                best_model_accuracy = epoch_validation_accuracy
                best_model_epoch = epoch
                #save_path = saver.save(sess, SAVE_PATH)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stop_epoch_limit:
                #print("Validation accuracy did not improve for %i epochs, stopping" % EARLY_STOPP_EPOCH_LIMIT)
                break

        print("learning rate=%.4f, weight decay=%.4f, accuracy: %.3f (epoch %i)" % (self.learning_rate, self.weight_decay, best_model_accuracy, best_model_epoch))
        return best_model_accuracy

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def test(self, test_minibatchgen):
        test_accuracies = []
        for i in range(0, test_minibatchgen.nbatches()):
            batch_data, batch_labels, _ = test_minibatchgen.batch(i)
            batch_labels_one_hot=np.eye(10)[batch_labels]

            test_accuracy=self.sess.run(self.accuracy, feed_dict={self.x: batch_data, self.y_: batch_labels_one_hot})

            test_accuracies.append(test_accuracy)

        test_accuracy = np.mean(test_accuracies)
        return test_accuracy