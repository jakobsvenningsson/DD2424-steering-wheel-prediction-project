import pandas as pd
import tensorflow as tf
import numpy as np
import os
import cv2
import numpy as np
import sys

def main(): 
    print("Running main")
    labelFile = sys.argv[1]         # suggest renaming to labelDir  
    imageData = sys.argv[2]         # suggest renaming to imageDir
    n_classes = 3                   # left, straight, right
    keep_prob = tf.placeholder(tf.float32)
    batch_size = 50 

    df = pd.read_csv(str(labelFile) + "/final_example.csv");    # labels
    rows, cols = df.shape           # get dimensions of label file

    # Assign a label to each image
    y_data = np.zeros((rows, n_classes), dtype=float) # initialize y to {0} 
    for index in range(rows):   
        angle = df.loc[index, 'steering_angle']
        if angle > 0.1:
            y_data[index, 2] = 1
        elif angle < -0.1:
            y_data[index, 0] = 1
        else:
            y_data[index, 1] = 1

    print(y_data)

    # Load image data
    images = os.listdir(str(imageData));    # get all image paths
    images = images[50:2100]                # truncate selected images
    x_data = np.array([cv2.imread(os.path.join(str(imageData), img)).flatten() for img in images if img.endswith(".jpg")], dtype=np.float32)
    x_train = x_data[50:, :]
    y_train = y_data[50:, :]
    x_test = x_data[0: 50, :]
    y_test = y_data[0: 50, :]
    x = tf.placeholder("float", [None, 640 * 480 * 3] )
    y = tf.placeholder("float", [None, n_classes])


    # Evaluate network
    prediction = cnn(x, keep_prob, n_classes)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    train_neural_network(x_train, y_train, batch_size, y, optimizer, cost)


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


if __name__ == "__main__":
    main()