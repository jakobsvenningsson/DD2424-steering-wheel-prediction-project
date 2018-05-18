import tensorflow as tf
#import pandas as pd
import numpy as np
import os
import numpy as np
import sys

def input_parser(img_path, label):
    """
    Parse input images and labels.
    Set grayscale for every image and resize to 240x320.
    Generate one-hot representations of labels

    :args:relative path to image and label
    :return:img_decoded
    :return:one_hot
    """
    #one_hot = tf.one_hot(label, 3)
    img_file = tf.read_file(img_path)
    #img_file = tf.read_file("/data2/center6/" + img_path + ".jpg")
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.resize_images(img_decoded, [60, 80])
    return img_decoded, label

def _parse_line(line):
    """
    Parse line from csv file.
    Extract frame id which is the image path name
    Extract the corresponding label

    :args: line - single csv row
    :return:path (or filename)
    :return:label
    """
    fields = tf.decode_csv(line, [[""], [0.0], [0]])
    features = dict(zip(["frame_id", "steering_angle", "label"],fields))
    path = features.pop('frame_id')
    angle = features.pop("steering_angle")
    #label = features.pop('label')
    return path, angle

def cnn(x, n_classes, keep_prob):
    """

    """
    #xavier_init = tf.random_normal
    xavier_init = tf.keras.initializers.he_normal()
    weights = {
            'W_conv1': tf.Variable(xavier_init([5, 5, 3, 32])),
            'W_conv2': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_conv3': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_conv4': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_fc_1': tf.Variable(xavier_init([15 * 20 * 32, 15 * 20 * 32])),
            'W_fc_2': tf.Variable(xavier_init([15 * 20 * 32, 1])),
            'W_fc_3': tf.Variable(xavier_init([15 * 20 * 32, 1])),
            #'out': tf.Variable(xavier_init([1024, 1]))
    }


    biases = {
            'b_conv1': tf.Variable(xavier_init([32])),
            'b_conv2': tf.Variable(xavier_init([32])),
            'b_conv3': tf.Variable(xavier_init([32])),
            'b_conv4': tf.Variable(xavier_init([32])),
            'b_fc_1': tf.Variable(xavier_init([15 * 20 * 32])),
            'b_fc_2': tf.Variable(xavier_init([1])),
            'b_fc_3': tf.Variable(xavier_init([1])),
            #'out': tf.Variable(xavier_init([1]))
    }

    conv1 = tf.nn.leaky_relu(conv2d(x, weights['W_conv1']) +  biases['b_conv1'])
    conv2 = tf.nn.leaky_relu(conv2d(conv1, weights['W_conv2']) +  biases['b_conv2'])
    conv2 = maxpool2d(conv1)

    conv3 = tf.nn.leaky_relu(conv2d(conv2, weights['W_conv3'] + biases['b_conv3']))
    conv4 = tf.nn.leaky_relu(conv2d(conv3, weights['W_conv4'] + biases['b_conv4']))
    conv4 = maxpool2d(conv2)

    fc = tf.reshape(conv4, [-1, 15 * 20 * 32])
    fc1 = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc_1'] + biases['b_fc_1']))
    fc2 = tf.nn.leaky_relu(tf.matmul(fc1, weights['W_fc_2'] + biases['b_fc_2']))
    #fc3 = tf.nn.leaky_relu(tf.matmul(fc2, weights['W_fc_3'] + biases['b_fc_3']))
    #fc = tf.nn.dropout(fc, keep_prob)
    #output = tf.matmul(fc, weights['out'] + biases['out'])
    return fc2

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def train_neural_network(y, optimizer, cost):
    _out_final_accuracy = open("final_acc_test.out", "w")
    print("Training starts.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        validation_writer = tf.summary.FileWriter('./logs/validation', sess.graph)
        for epoch in range(hm_epochs):
            sess.run(ds_train_iterator.initializer)
            epoch_accuracy = 0
            while True:
                try:
                    elem = sess.run(train_next_element)
                    merge = tf.summary.merge_all()
                    _, c = sess.run([optimizer, cost], feed_dict={x: elem[0], y: elem[1], keep_prob: 0.8})
                    print(c)
                    _, epoch_accuracy, summary, mean_absolute_err = sess.run([avg_acc, avg_acc_update, merge, mean_absolute_error_update], feed_dict={x: elem[0], y: elem[1], keep_prob: 0.8})
                    train_writer.add_summary(summary, epoch)
                    print("loss: ", mean_absolute_err)
                except tf.errors.OutOfRangeError:
                    """
                    s = 5
                    epoch_validation_accuracy = 0
                    sess.run(ds_validation_iterator.initializer)
                    for i in range(s):
                        try:
                            elem_validation = sess.run(validation_next_element)
                            merge = tf.summary.merge_all()
                            _, c = sess.run([optimizer, cost], feed_dict={x: elem_validation[0], y: elem_validation[1], keep_prob: 1.0})
                            _, epoch_validation_accuracy, summary, mean_err = sess.run([avg_acc, avg_acc_update, merge, mean_error_update], feed_dict={x: elem_validation[0], y: elem_validation[1], keep_prob: 1.0})
                            predictions = prediction.eval(feed_dict = {x: elem_validation[0], keep_prob: 1.0})
                            validation_writer.add_summary(summary, epoch)
                            if i == 0:
                                print(predictions)
                        except tf.errors.OutOfRangeError:
                            print('ERROR Accuracy:',accuracy/s)
                    print("End of epoch.")
                    print("Train accuracy; ", epoch_accuracy)
                    print("validation accuracy: ", epoch_validation_accuracy)
                    """
                    break
            print('Epoch', epoch, 'completed out of',hm_epochs)
        """
        s = 10
        epoch_test_accuracy = 0
        for i in range(s):
            try:
                sess.run(ds_test_iterator.initializer)
                elem_test = sess.run(test_next_element)
                _, epoch_test_accuracy = sess.run([avg_acc, avg_acc_update], feed_dict={x: elem_test[0], y: elem_test[1], keep_prob: 1.0})

            except tf.errors.OutOfRangeError:
                print("OUT OF RANGE")
        print("TEST ACCURACY: ", epoch_test_accuracy)
        _out_final_accuracy.write(str(epoch_test_accuracy) + "\n")

        """    
"""
:: START OF EXECUTION ::
"""

"""
Initilize phase

    batch_size: size of consecutive elements into (batching)
    n_classes : left = 0, center = 1, right = 2.

"""
batch_size = 256
n_classes = 1
image_width = 320
image_height = 240
hm_epochs = 80
print("Setting up")

ds_test = tf.data.TextLineDataset("test.csv").skip(1)
ds_test = ds_test.map(_parse_line)
ds_test = ds_test.map(input_parser)
ds_test = ds_test.batch(batch_size)
ds_test_iterator = ds_test.make_initializable_iterator();
test_next_element = ds_test_iterator.get_next()
print("test data setup completed")

ds_validation = tf.data.TextLineDataset("validation.csv").skip(1)
ds_validation = ds_validation.map(_parse_line)
ds_validation = ds_validation.map(input_parser)
ds_validation = ds_validation.batch(batch_size)
ds_validation_iterator = ds_validation.make_initializable_iterator();
validation_next_element = ds_validation_iterator.get_next()
print("Validation data setup completed")

ds_train = tf.data.TextLineDataset("train.csv").skip(1)
ds_train = ds_train.map(_parse_line)
ds_train = ds_train.map(input_parser)
ds_train = ds_train.shuffle(buffer_size=8000)
ds_train = ds_train.repeat(1)
ds_train = ds_train.batch(batch_size)
ds_train_iterator = ds_train.make_initializable_iterator();
train_next_element = ds_train_iterator.get_next()
print("Train data setup completed")



keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, 60, 80, 3])
y = tf.placeholder("float", [None])

prediction = cnn(x, n_classes, keep_prob)

# Compute softmax cross entropy between logits and labels.
# Mesures the probability error in descrete classification tasks in which
# the classes are mutually exclusive
# https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2
mean_squared_error = tf.losses.mean_squared_error(y, prediction[0])
#out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y)
#tf.summary.histogram("Predictions Histogram", out)

# LEGACY CODE
#tf.summary.scalar("Predictions Scalar", out)
#correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#accs = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
tf.metrics.accuracy(labels, predictions):
Calculates how often predictions matches labels (y)

https://www.tensorflow.org/api_docs/python/tf/metrics/accuracy
"""
avg_acc, avg_acc_update = tf.metrics.accuracy(prediction[0], y)

# LEGACY CODE
#tf.summary.histogram("Accuracy Histogram", update_)

"""
tf.summary.scalar(name, real numeric tensor):
Outputs a Summary protocol buffer containing a single scalar value

https://www.tensorflow.org/api_docs/python/tf/summary/scalar
"""
tf.summary.scalar("Accuracy Scalar", avg_acc)
#class_weights = tf.constant([0.8, 0.8, 1.0])
#out = tf.nn.weighted_cross_entropy_with_logits(logits=prediction, targets=y, pos_weight=class_weights)'
"""
tf.reduce_mean(input tensor)
Computes the mean of elements across dimensions of a tensor.

https://www.tensorflow.org/api_docs/python/tf/reduce_mean
"""
mean_absolute_error, mean_absolute_error_update = tf.metrics.mean_absolute_error(y, prediction[0])
cost = tf.reduce_mean(mean_squared_error)
tf.summary.scalar("Mean Absolute Error", mean_absolute_error)
tf.summary.scalar("Mean Squared Error", mean_squared_error)



"""
Optimizer

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.
"""
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
train_neural_network(y, optimizer, cost)
