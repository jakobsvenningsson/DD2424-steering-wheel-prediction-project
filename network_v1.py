import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import numpy as np
import sys

def input_parser(img_path, label):
    one_hot = tf.one_hot(label, 3)
    img_file = tf.read_file("drive/app/data/Ch2_001/center/" + img_path + ".jpg")
    #img_file = tf.read_file("data/Ch2_001/center/" + img_path + ".jpg")
    img_decoded = tf.image.decode_image(img_file, channels=3)
    return img_decoded, one_hot

def _parse_line(line):
    fields = tf.decode_csv(line, [[""], [0.0], [0]])
    features = dict(zip(["frame_id", "steering_angle", "label"],fields))
    #label = features.pop('steering_angle')
    path = features.pop('frame_id')
    label = features.pop('label')
    return path, label

def cnn(x, n_classes, keep_prob):

    #with tf.device("/device:gpu:0"):        
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 32])),
        'W_conv3': tf.Variable(tf.random_normal([5, 5, 32, 32])),
        'W_fc': tf.Variable(tf.random_normal([80 * 60 * 32, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
        }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([32])),
        'b_conv3': tf.Variable(tf.random_normal([32])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
        }

    # Convert picture to grayscale to reduce size
    # This should be turned off when we train on PDC.
    # x = tf.image.rgb_to_grayscale(x)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) +  biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'] + biases['b_conv2']))
    conv2 = maxpool2d(conv2)
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3'] + biases['b_conv3']))
    conv3 = maxpool2d(conv3)
    fc = tf.reshape(conv3, [-1, 80 * 60 * 32])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'] + biases['b_fc']))
    fc = tf.nn.dropout(fc, keep_prob)
    output = tf.matmul(fc, weights['out'] + biases['out'])

    return output

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def train_neural_network(batch_size, y, optimizer, cost):
    hm_epochs = 10
    print("Training starts.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(ds_test_iterator.initializer)
        sess.run(ds_train_iterator.initializer)
        elem_test = sess.run(test_next_element)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            count = 0
            while True:
                try:
                    elem = sess.run(train_next_element)
                    _, c = sess.run([optimizer, cost], feed_dict={x: elem[0], y: elem[1], keep_prob: 0.8})
                    epoch_loss += c
                    count += 1
                    if count % 7 == 1:
                        predictions = prediction.eval(feed_dict = {x: elem[0], keep_prob: 0.8})
                        print(predictions)
                        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                        print('Accuracy:',accuracy.eval({x: elem_test[0], y: elem_test[1], keep_prob: 1.0}))
                except tf.errors.OutOfRangeError:
                    print("End of epoch.")
                    break

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            try:
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',accuracy.eval({x: elem_test[0], y: elem_test[1], keep_prob: 1.0 }))
            except tf.errors.OutOfRangeError:
                print("failed to get test data.")
                break


ds_test = tf.data.TextLineDataset("drive/app/data/Ch2_001/final_example_test.csv").skip(1)
#ds_test = tf.data.TextLineDataset("data/Ch2_001/final_example_test.csv").skip(1)
ds_test = ds_test.map(_parse_line)
ds_test = ds_test.map(input_parser)
ds_test = ds_test.batch(64)
ds_test_iterator = ds_test.make_initializable_iterator()
test_next_element = ds_test_iterator.get_next()

ds_train = tf.data.TextLineDataset("drive/app/data/Ch2_001/final_example_train.csv").skip(1)
#ds_train = tf.data.TextLineDataset("data/Ch2_001/final_example_train.csv").skip(1)
ds_train = ds_train.map(_parse_line)
ds_train = ds_train.map(input_parser)
ds_train = ds_train.shuffle(buffer_size=200)
ds_train = ds_train.repeat(10)
ds_train = ds_train.batch(64)
ds_train_iterator = ds_train.make_initializable_iterator();
train_next_element = ds_train_iterator.get_next()
n_classes = 3

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, 480, 640, 3])
y = tf.placeholder("float", [None, n_classes])

prediction = cnn(x, n_classes, keep_prob)
out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y)
cost = tf.reduce_mean(out)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
train_neural_network(50, y, optimizer, cost)
