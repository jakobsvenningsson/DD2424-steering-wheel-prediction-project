import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import numpy as np
import sys

filename_queue = tf.train.string_input_producer(["data/Ch2_001/final_examples.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[1], [1]]
col1, col2 = tf.decode_csv(value, record_defaults=record_defaults)


def input_parser(img_path, label):
    print(img_path)
    print(label)
#    print(n_classes)
    one_hot = tf.one_hot(label, n_classes)
#    img_file = tf.read_file("data/Ch2_001/center/" + img_path + ".jpg")
#    img_decoded = tf.image.decode_image(img_file, channels=1)
#    print("DECODE")
    return img_path, one_hot


#labelFile = sys.argv[1]
#imageData = sys.argv[2]
#n_classes = 3
#keep_prob = tf.placeholder(tf.float32)
#batch_size = 50 

#filename_queue = tf.train.string_input_producer(["data/Ch2_001/final_examples.csv"])
#reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)
#record_defaults = [[1], [1]]
#col1, col2 = tf.decode_csv(
#value, record_defaults=record_defaults)
#print(col1)
#print(col2)
#df = pd.read_csv(str(labelFile) + "/final_example.csv");
#rows, cols = df.shape

# Assign a label to each image
#l = np.zeros((rows), dtype=float)
#for index in range(rows):
#    angle = df.loc[index, 'steering_angle']
#    if angle > 0.1:
#        l[index] = 2
#    elif angle < -0.1:
#        l[index] = 0
#    else:
#        l[index] = 1
#
#im = df["frame_id"]

#train_labels = tf.constant(l[50: ])
#train_images = tf.constant(im[50: ])
#val_labels = tf.constant(l[50: ])
#val_images = tf.constant(im[50:])

#tr_data = tf.data.Dataset.from_tensor_slices((im[50: ], l[50: ]))
#val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
#tr_data = tr_data.map(input_parser)
#val_data = val_data.map(input_parser)
#train_iterator = tr_data.make_initializable_iterator();
#validation_init_op = val_data.make_initializable_iterator()
#next_element = train_iterator.get_next()

# Load image data
#images = os.listdir(str(imageData));
#images = images[50:2100]
#x_data = np.array([cv2.imread(os.path.join(str(imageData), img)).flatten() for img in images if img.endswith(".jpg")], dtype=np.float32)
#x_train = x_data[50:, :]
#y_train = y_data[50:, :]
#x_test = x_data[0: 50, :]
#y_test = y_data[0: 50, :]

#x = tf.placeholder("float", [None, 640 * 480 * 3] )
#y = tf.placeholder("float", [None, n_classes])
"""
with tf.Session() as sess:

    sess.run(train_iterator.initializer, feed_dict={})
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
"""
def cnn(x, keep_prob, n_classes):

    #with tf.device("/device:gpu:0"):        
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
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

    x = tf.reshape(x, shape=[-1, 640, 480, 3])
    # Convert picture to grayscale to reduce size
    # This should be turned off when we train on PDC.
    x = tf.image.rgb_to_grayscale(x)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) +  biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'] + biases['b_conv2']))
    conv2 = maxpool2d(conv2)
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3'] + biases['b_conv3']))
    conv3 = maxpool2d(conv3)
    fc = tf.reshape(conv3, [-1, 80 * 60 * 32])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'] + biases['b_fc']))
#    fc = tf.nn.dropout(fc, keep_prob)
    output = tf.matmul(fc, weights['out'] + biases['out'])

    return output

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def train_neural_network(x_train, y_train, batch_size, y, optimizer, cost):
    hm_epochs = 10
    print("Training starts.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            rows, cols = x_train.shape
            for i in range(int(rows/batch_size) - 1):
                print("updated")
                print(i * batch_size,  (i  + 1) * batch_size)
                x_batch = x_train[(i) * batch_size:  (i + 1) * batch_size, :]
                y_batch = y_train[(i) * batch_size:  (i + 1) * batch_size, :]
                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})#, keep_prob: 0.8})
                epoch_loss += c
                print(epoch_loss)
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: x_test, y: y_test}))

#prediction = cnn(x, keep_prob, n_classes)
#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
#optimizer = tf.train.AdamOptimizer().minimize(cost)

#train_neural_network(x_train, y_train, batch_size, y, optimizer, cost)
