
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

def dogClassifier( X, y, alpha = 0.001, b1 = 0.9, b2 = 0.999, epsilon = 1e-08 ):
    """Creates all objects needed for training the dog classifier."""

    heInit = tf.variance_scaling_initializer()

    with tf.name_scope( "cnn" ):

        conv1 = tf.layers.conv2d( X, filters = 4, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv1", kernel_initializer = heInit )
        pool1 = tf.layers.max_pooling2d( conv1, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool1" )

        conv2 = tf.layers.conv2d( pool1, filters = 8, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv2", kernel_initializer = heInit )
        pool2 = tf.layers.max_pooling2d( conv2, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool2" )

        conv3 = tf.layers.conv2d( pool2, filters = 16, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv3", kernel_initializer = heInit )
        pool3 = tf.layers.max_pooling2d( conv3, pool_size = 2, strides = 1, padding = "valid",
                                         name = "pool3" )

        conv4 = tf.layers.conv2d( pool3, filters = 2, kernel_size = 1, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  name = "conv4", kernel_initializer = heInit )

        flat = tf.layers.flatten( conv4 )

        logits = tf.layers.dense( flat, 120, name = "output", kernel_initializer = heInit )

    with tf.name_scope("loss"):
        crossEnt = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = y, logits = logits)
        loss = tf.reduce_mean( crossEnt, name = "loss" )

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )

    with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer( learning_rate = alpha,
                                      beta1 = b1, beta2 = b2,
                                      epsilon = epsilon )
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

def trainModel( trainX, trainY, valX, valY, params, saveModel = False ):

    #valLoadedX, valLoadedY = genData( valX, valY, size = 200 )

    parameters = params[ "params" ]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (None, 200, 200, 3), name = "X")
    y = tf.placeholder(tf.int32, shape = (None), name = "y")

    loss, training, accuracy, lossSummary, init, saver = dogClassifier( X, y, **parameters )

    nEpochs = 5000
    batchSize = params[ "batchSize" ]

    loVal = 10000
    patience = 0

    step = 0

    with tf.Session() as sess:

        init.run()

        #tls = [ loss.eval( feed_dict = { X : trainX, y : trainY["breed"].values } ) ]
        tls = [ 10000 ]
        #vls = [ loss.eval( feed_dict = { X : valLoadedX, y : valLoadedY } ) ]
        vls = [ batchEval( X, y, valX, valY, batchSize, loss ) ]

        for epoch in range(nEpochs):
            for batchX, batchY in genBatch( trainX, trainY, batchSize, imgSize = 200 ):

                sess.run( training, feed_dict = { X : batchX, y : batchY } )
                step += 1

                if ( step % 50 == 0 ):
                    #valLoss = loss.eval( feed_dict = { X : valLoadedX, y : valLoadedY } )
                    valLoss =  batchEval( X, y, valX, valY, batchSize, loss )

                    print("Step:", step, "valLoss:", valLoss )

                    if ( valLoss < loVal ):
                        loVal = valLoss

                        if (saveModel):
                            saver.save( sess, "./best/mnist-best.ckpt" )

            trainLoss = loss.eval( feed_dict = { X : batchX, y : batchY } )
            #trainLoss =  batchEval( batchX, batchY, batchSize, loss )
            #valLoss   = loss.eval( feed_dict = { X : valLoadedX, y : valLoadedY } )
            valLoss   =  batchEval( X, y, valX, valY, batchSize, loss )

            tls.append( trainLoss )
            vls.append( valLoss )

            if ( valLoss < loVal ):
                loVal = valLoss
                patience = 0

            else:
                patience += 1

            if ( patience >= 20):
                break

        if (saveModel):
            print()
            print("***")
            #print("Final model validation accuracy:", accuracy.eval(feed_dict = { X : valX, y : valY }) )
            print("Final model validation accuracy:", batchEval( X, y, valX, valY, batchSize, accuracy ) )
            print("***")
            print()

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

