
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from dogFunctions import genData, genBatch


def inceptionBlock( X, nPath1, nPath2_1, nPath2_2, nPath3_1, nPath3_2, nPath4 ):
    """Creates an inception block."""

    with tf.name_scope( "inceptionBlock" ):
        path1 = tf.layers.conv2d( X, filters = nPath1, kernel_size = 1, strides = 1,
                                  padding = "SAME", activation = tf.nn.relu  )

        in2   = tf.layers.conv2d( X,   filters = nPath2_1, kernel_size = 1, strides = 1,
                              padding = "SAME", activation = tf.nn.relu  )
        path2 = tf.layers.conv2d( in2, filters = nPath2_2, kernel_size = 3, strides = 1,
                              padding = "SAME", activation = tf.nn.relu )

        in3   = tf.layers.conv2d( X,   filters = nPath3_1, kernel_size = 1, strides = 1,
                              padding = "SAME", activation = tf.nn.relu )
        path3 = tf.layers.conv2d( in3, filters = nPath3_2, kernel_size = 5, strides = 1,
                              padding = "SAME", activation = tf.nn.relu )

        in4   = tf.layers.max_pooling2d( X, pool_size = 2, strides = 1, padding = "SAME",
                                     name = "p4.1" )
        path4 = tf.layers.conv2d( in4, filters = nPath4, kernel_size = 1, strides = 1,
                              padding = "SAME", activation = tf.nn.relu  )

        return tf.concat( [path1, path2, path3, path4], axis = 3 )

def dogClassifier( X, y, trn, alpha = 0.001, b = 0.9, bnm = 0.99 ):
    """Creates all objects needed for training the dog classifier."""

    heInit = tf.variance_scaling_initializer()

    with tf.name_scope( "cnn" ):

        conv1 = tf.layers.conv2d( X, filters = 4, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv1", kernel_initializer = heInit )
        pool1 = tf.layers.max_pooling2d( conv1, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool1" )

        conv2 = tf.layers.conv2d( pool1, filters = 6, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv2", kernel_initializer = heInit )
        pool2 = tf.layers.max_pooling2d( conv2, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool2" )

        conv3 = tf.layers.conv2d( pool2, filters = 8, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv3", kernel_initializer = heInit )
        pool3 = tf.layers.max_pooling2d( conv3, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool3" )

        conv4 = tf.layers.conv2d( pool3, filters = 10, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv4", kernel_initializer = heInit )
        pool4 = tf.layers.max_pooling2d( conv4, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool4" )

        flat = tf.layers.flatten( pool4 )
        bn1 = tf.layers.batch_normalization( flat, training = trn, momentum = bnm )

        fc1 = tf.layers.dense( flat, 240, name = "fc1", kernel_initializer = heInit,
                               activation = tf.nn.elu )
        fc2 = tf.layers.dense( fc1, 120, name = "fc2", kernel_initializer = heInit,
                               activation = tf.nn.elu )

        logits = tf.layers.dense( fc1, 120, name = "output", kernel_initializer = heInit )

    with tf.name_scope("loss"):
        crossEnt = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = y, logits = logits)
        loss = tf.reduce_mean( crossEnt, name = "loss" )

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )

    with tf.name_scope("train"):
        opt = tf.train.MomentumOptimizer( learning_rate = alpha, momentum = b,
                                          use_nesterov = False )

        training = opt.minimize( loss )
        lossSummary = tf.summary.scalar("crossEntropy", loss)

    with tf.name_scope("utility"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return loss, training, accuracy, lossSummary, init, saver

def batchEval( X, y, allX, allY, batchSize, function ):

    count = 0
    temp = 0

    for start in range(0, len(allX) - 1, batchSize):
        bX, bY = genData( allX[ start : start + batchSize ], allY, size = 200 )
        n = len(bX)
        count += n

        temp += function.eval( feed_dict = { X : bX, y : bY } ) * n

    return temp / count

def trainModel( trainX, valX, labels, params, saveModel = False ):

    parameters = params[ "params" ]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (None, 200, 200, 3), name = "X")
    y = tf.placeholder(tf.int32, shape = (None), name = "y")
    trn = tf.placeholder_with_default( False, shape = (), name = "trn" )

    loss, training, accuracy, lossSummary, init, saver = dogClassifier( X, y, trn, **parameters )

    #nEpochs = 1 #5000
    batchSize = params[ "batchSize" ]

    loVal = 10000
    patience = 0

    step = 0

    with tf.Session() as sess:

        init.run()

        tls = [ 10000 ]
        vls = [ batchEval( X, y, valX, labels, batchSize, loss ) ]

        #for epoch in range(nEpochs):
        while ( True ):
            for batchX, batchY in genBatch( trainX, labels, batchSize, imgSize = 200 ):

                sess.run( training, feed_dict = { X : batchX, y : batchY, trn : True } )
                step += 1

                if ( step % 25 == 0 ):
                    valLoss = batchEval( X, y, valX, labels, batchSize, loss )
                    valAcc  = batchEval( X, y, valX, labels, batchSize, accuracy )
                    trainLoss = loss.eval( feed_dict = { X : batchX, y : batchY } )

                    tls.append( trainLoss )
                    vls.append( valLoss )

                    print("Step: {0:>5d}, valLoss: {1:8.6f}, trainLoss: {2:8.6f}, valAcc: {3:5.4f}".format(step, valLoss, trainLoss, valAcc) )

                    if ( valLoss < loVal ):
                        loVal = valLoss
                        patience = 0

                        if (saveModel):
                            saver.save( sess, "./best/mnist-best.ckpt" )

                    else:
                        patience += 1

                    if ( patience >= 20):
                        break

        print("\n***")
        print("Final validation accuracy:", batchEval( X, y, valX, labels, batchSize, accuracy ) )
        print("***\n")

    return (loVal, tls, vls)

def crossValidation( trainX, trainY, k, params ):

    res = []
    kf = KFold( n_splits = k )

    for indTr, indVl in kf.split( trainX, trainY ):
        valLoss , tls, vls = trainModel( trainX[indTr], trainY,
                                         trainX[indVl], trainY, params )

        res.append(valLoss)

    return np.mean( res )

def hyperparameterSearch( trainX, trainY, paramsList, k ):

    bestLoss = 100000
    modelID = 0
    bestParams = None

    for params in paramsList:

        t1 = time.time()
        valLoss = crossValidation( trainX, trainY, k, params )
        t2 = time.time()

        if ( valLoss < bestLoss ):
            bestLoss = valLoss
            bestParams = params

        print("Done {0:4d} of {1:4d} in {2:4.1f}s".format(modelID + 1, len(paramsList), t2 - t1),
              "Validation loss:", valLoss )
        modelID += 1

    trX, valX, trY, valY = train_test_split( trainX, trainY, test_size = 500,
                                                   random_state = 123 )

    loVal, trHist, valHist = trainModel( trX, trY, valX, valY, bestParams, saveModel = True )

    return ( loVal, trHist, valHist, bestParams )

