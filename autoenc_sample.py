#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Spyder Editor

This example is based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb
"""

import tensorflow as tf
assert(tf.VERSION >= '1.0.0')
import numpy as np
import matplotlib as mplib
import json
import os
from httplib import HTTPSConnection

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
IMG_DIMS = [28, 28, 1]  #images are 28x28, single channel
KERNEL_SIZE = [5,5] #convolutional filter kernel size

def readMNISTDataset():
    return input_data.read_data_sets("MNIST_data", one_hot=True)

def getParameters():
    # Network parameters (available at the command line)
    flags = tf.app.flags
    flags.DEFINE_integer('instance', 0, "Instance index")
    flags.DEFINE_integer('training_epochs', 5, "Number of training epochs")
    flags.DEFINE_float('learning_rate', 0.01, "Learning rate")
    flags.DEFINE_integer('batch_size', 256, "batch size for learning")
    flags.DEFINE_integer('n_conv_1', 32, "number of filters at first convolutional hidden level")
    flags.DEFINE_integer('n_conv_2', 64, "number of filters at second convolutional hidden level")
    flags.DEFINE_integer('n_pool_1', 2, "size of pooling nodes at first pooling hidden level")
    flags.DEFINE_integer('n_pool_2', 2, "size of pooling nodes at second pooling hidden level")
    flags.DEFINE_integer('n_rec', 1024, "number of nodes at first reconstruction hidden level")
    flags.DEFINE_float('dropout', 0.4, "Dropout regularization probability")
    flags.DEFINE_string('log_dir', '/tmp/logs', "Directory to write log files in")
    return flags.FLAGS

def buildEncoder(x, isTraining, flags):
    """ 
    Builds the encoder network and returns outputs from the encoder layer
    """
    #input layer
    inputLayer = tf.reshape(x, [flags.batch_size, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2]])

    # Convolutional Layer #1
    # Computes FLAGS.n_conv_1 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, FLAGS.n_conv_1]
    # For multi-channel images, use tf.contrib.layers.conv2d_in_plane
    conv1 = tf.layers.conv2d(
            inputs=inputLayer,
            filters=flags.n_conv_1,
            kernel_size=KERNEL_SIZE,
            padding="SAME",
            activation=tf.nn.relu,
            name='v1')

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, FLAGS.n_conv_1]
    # Output Tensor Shape: [batch_size, 14, 14, FLAGS.n_conv_1]
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=flags.n_pool_1,
            strides=2)
    
    # Dropout layer
    dropout = tf.layers.dropout(
            inputs = pool1, 
            rate = flags.dropout, 
            training = isTraining)
    
    # Convolutional Layer #2
    # Computes FLAGS.n_conv_2 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, FLAGS.n_conv_1]
    # Output Tensor Shape: [batch_size, 14, 14, FLAGS.n_conv_2]
    conv2 = tf.layers.conv2d(
            inputs=dropout,
            filters=flags.n_conv_2,
            kernel_size=KERNEL_SIZE,
            padding="SAME",
            activation=tf.nn.relu,
            name='v2')
    
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2, 
            pool_size=flags.n_pool_2,
            strides=2)

    filter_summary_1 = tf.contrib.layers.summaries.summarize_weights(conv1)
    filter_summary_2 = tf.contrib.layers.summaries.summarize_weights(conv2)
    enc_summary = tf.summary.merge([filter_summary_1, filter_summary_2])

    #Flatten both convolutional layer features & merge as encoder output
    flat1 = tf.contrib.layers.flatten( pool1 )
    flat2 = tf.contrib.layers.flatten( pool2 )
    encOutput = tf.concat([flat1, flat2], 1)

    return (encOutput, enc_summary)    

def buildDecoder(encodedInput, isTraining, flags, nOutputs):
    """ 
    Builds the decoder network and returns the output from the decoder layer
    """
    #Reconstruction layer #1, uses features from both pooled layers
    rec1 = tf.layers.dense(
            inputs = encodedInput,
            units = flags.n_rec,
            activation = tf.nn.relu)
    #Dropout regularization
    dropout = tf.layers.dropout(
            inputs = rec1, 
            rate = flags.dropout, 
            training = isTraining)
    
    #Reconstruction layer #2
    rec2 = tf.layers.dense(
            inputs = dropout,
            units = nOutputs,
            activation = tf.nn.relu,
            name = 'output')
    
    return (rec2, "")

def buildGraph(mnist, flags):
    """
    Bulds the graph
    """
    N_INPUT = mnist.train.images.shape[1] # (linearized img shape: 28*28)
    N_CATS = mnist.train.labels.shape[1]

    # tf Graph input (only pictures)
    x = tf.placeholder("float", [None, N_INPUT], name="input")
    isTraining = tf.placeholder("bool", [], name="is_training");
    
    # ENCODING LAYERS
    enc, enc_summary = buildEncoder(x, isTraining, flags)
    out, dec_summary = buildDecoder(enc, isTraining, flags, N_INPUT)

    # Learning criteria
    cost = tf.reduce_mean(tf.pow(x - out, 2))
    optimizer = tf.train.RMSPropOptimizer(flags.learning_rate).minimize(cost)

    graphSummary = tf.summary.merge([enc_summary, dec_summary])
    graph = {'input': x, 
             'output': out, 
             'cost': cost, 
             'optimizer': optimizer}
    return (graph, graphSummary)
    
def trainAutoencoder(tfSession, dataset, graph, summary, flags):
    """
    Trains a built graph. The variables should be previously initialized
    """
    # Define loss and optimizer, minimize the squared error
    x = graph['input']
    cost = graph['cost']
    optimizer = graph['optimizer']
    # Init logger
    costSummary = tf.summary.scalar('cost', cost)
    summarizer = tf.summary.FileWriter(flags.log_dir, graph=tfSession.graph)
    
    # Training cycle
    isTraining = tfSession.graph.get_tensor_by_name("is_training:0")
    nBatches = int(dataset.num_examples/flags.batch_size)
    c = np.inf
    for epoch in range(1, 1+flags.training_epochs):
        # Loop over all batches
        for i in range(nBatches):
            step = (epoch - 1) * nBatches + i
            # get next batch of images. 
            # since we're minimizing reconstruction errors, labels are ignored
            xBatch, _ = dataset.next_batch(flags.batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, costSummaryStr, summaryStr, c = tfSession.run(
                    [optimizer, costSummary, summary, cost], 
                    feed_dict={x: xBatch, isTraining: True})
            summarizer.add_summary(costSummaryStr, global_step=step)
        # Display logs per epoch step
        print "Epoch: {0:3d}, cost = {1:.6f}".format(epoch, c)
        if (0 == (epoch % 5)):
            summarizer.add_summary(summaryStr, global_step=step)
    summarizer.close()
    return c
    
def testModel(tfSession, dataset, graph, flags, nImgs=10):
    """
    Tests the models by passing images & displaying reconstructions
    """
    # Applying encoder and decoder over test set
    x = graph['input']
    isTraining = tfSession.graph.get_tensor_by_name('is_training:0')
    v1filters = tfSession.graph.get_collection('trainable_variables', 'v1')[0]
    v2filters = tfSession.graph.get_collection('trainable_variables', 'v2')[0]
    output, v1imgs, v2imgs = tfSession.run(
            [graph['output'], v1filters, v2filters], 
            feed_dict={x: dataset.images[:flags.batch_size], isTraining: False})
    # Compare original images with their reconstructions
    f, a = mplib.pyplot.subplots(2, nImgs, figsize=(nImgs, 2))
    for i in range(nImgs):
        a[0][i].imshow(np.reshape(mnist.test.images[i], IMG_DIMS[0:2]))
        a[1][i].imshow(np.reshape(output[i], IMG_DIMS[0:2]))
    f.show()
    
    #Figure out how to 'name' the convolutional layers and get their variables
    #v1imgs = v1filters.eval(tfSession)
    #v2imgs = v2filters.eval(tfSession)
    
    fV1, aV1 = mplib.pyplot.subplots(8, flags.n_conv_1/8)
    assert( v1imgs.shape[-1] == flags.n_conv_1 )
    for i in range(flags.n_conv_1):
        aV1[i%8][i/8].imshow(v1imgs[:,:,0,i], cmap='gray')
#        aV1[i%8][i/8].imshow(v1imgs[:,:,0,i], cmap='gray', interpolation='nearest')
    fV1.show()
    
    fV2, aV2 = mplib.pyplot.subplots(8, 8)
    for i in range(64):
        aV2[i%8][i/8].imshow(v2imgs[:,:,0,i], cmap='gray')
    fV2.show()

def postSlackMessage(token, text, channel=None):
    """
    Used to post a message to my slack channel when training is complete
    See https://api.slack.com/incoming-webhooks
    """
    server = 'https://hooks.slack.com'
    url = server + '/services/' + token
    payload = {'text': text, 
               'icon_emoji': ':docker:',
               'username': 'docker-bot'}
    if channel is not None:
        payload['channel'] = channel
    header = {'Content-type': 'application/json'}
    conn = HTTPSConnection(server + ':' + str(HTTPSConnection.default_port))
    conn.request('POST', url, json.dumps(payload), header)
    response = conn.getresponse()
    if response.reason != 'OK':
        print 'Unable to post text to Slack\n{0}: {1}'.format(response.status, response.reason)
    conn.close()

    
def classifier(x, flags, weights):
    #convolution layers
    x = tf.reshape(x, [flags.batch_size, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2]])
    conv1 = tf.nn.relu(tf.nn.conv2d(x, weights['filter1'], strides = [1,1,1,1], padding = "SAME" ))

    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=flags.n_pool_1,
            strides=2)
    
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights['filter2'], strides = [1,1,1,1], padding = "SAME" ))

    
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2, 
            pool_size=flags.n_pool_2,
            strides=2)

    #fully connected dense layer
    pool2 = tf.contrib.layers.flatten(pool2)
    class1 = tf.layers.dense(
              inputs = pool2,   
              units = 1024,
              activation = tf.nn.relu
             )
    
    class2 = tf.layers.dense(
            inputs = class1,
            units = 10
            )
    
    return class2


if __name__ == '__main__':
    #Read dataset & parameters
    mnist = readMNISTDataset()
    flags = getParameters()


    #Build graph
    graph, summary = buildGraph(mnist, flags)
    
    #start an interactive session
    sess = tf.InteractiveSession()

    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    cost = -1
    for majorEpoch in range(2):
        print 'Major Epoch {0}:'.format(majorEpoch+1)
        #Train model    
        cost = trainAutoencoder(sess, mnist.train, graph, summary, flags)
        #Test model
        testModel(sess, mnist.test, graph, flags)

    print 'Optimization Finished!'
    saver = tf.train.Saver({"filter1": sess.graph.get_collection('trainable_variables', 'v1')[0], 
                       "filter2": sess.graph.get_collection('trainable_variables', 'v2')[0]})
    print os.path
    savepath = os.path.join(".", "conv_featurelearning")
    modelPath = saver.save(sess, savepath)
    print 'Model saved to {0}.'.format(modelPath)

    
    
    
    
    
    #### classifier ########
    batch_size = 256
    
    x = tf.placeholder("float", [None, N_INPUT], name="input")
    y = tf.placeholder("float", [None, 10], name = "label")
    isTraining = tf.placeholder("bool", [], name="is_training");

    weights = {
        'filter1': tf.Variable(tf.random_normal([5,5,1,32]), name = "filter1"),
        'filter2': tf.Variable(tf.random_normal([5,5,32,64]), name = "filter2")
    }
    
    y_pred = classifier(x, flags, weights)


    # Learning criteria
    costc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    optimizer2 = tf.train.RMSPropOptimizer(flags.learning_rate).minimize(costc)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #start an interactive session
    sess2 = tf.InteractiveSession()

    # Initialize the variables
    sess2.run(tf.global_variables_initializer())
    print weights['filter1'].eval()
    saver.restore(sess2, savepath)
    print weights['filter1'].eval()

    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(3):
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            _,c,acc = sess2.run([optimizer2,costc, accuracy], feed_dict={x: batch_x, y: batch_y,
                                       isTraining: True})
            print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c),
              "accuracy=", "{:.5f}".format(acc))
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:", \
    sess2.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                              })

    
    ######### end of classifier code ##########
    
    
    
    message = """
    Training complete for Autoencoder instance {0}.
    Final cost = {1}.
    Parameters:
    """.format(flags.instance, cost)

    params = flags.__dict__['__flags']
    for key in params.keys():
        message += '\t{0:16}: {1}\n'.format(key, params[key])
    with open('keys.json') as keyFile:
        keys = json.loads(keyFile.read())
    slackToken = keys['slack_token']
    postSlackMessage(slackToken, message)
