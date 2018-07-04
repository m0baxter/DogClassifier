
import tensorflow as tf
from dogFunctions import genData, genBatch


def inceptionBlock( X, nPath1, nPath2_1, nPath2_2, nPath3_1, nPath3_2, nPath4 ):
    """Creates an inception block."""

    tf.name_scope( "inceptionBlock" ):
        path1 = tf.layers.conv2d( X, filters = np1, kernel_size = 1, strides = 1,
                                  padding = "SAME", activation = tf.nn.relu, name = "p1" )

        in2   = tf.layers.conv2d( X,   filters = np2_1, kernel_size = 1, strides = 1,
                              padding = "SAME", activation = tf.nn.relu, name = "p2.1" )
        path2 = tf.layers.conv2d( in2, filters = np2_2, kernel_size = 3, strides = 1,
                              padding = "SAME", activation = tf.nn.relu, name = "p2.2" )

        in3   = tf.layers.conv2d( X,   filters = np3_1, kernel_size = 1, strides = 1,
                              padding = "SAME", activation = tf.nn.relu, name = "p3.1" )
        path3 = tf.layers.conv2d( in3, filters = np3_2, kernel_size = 5, strides = 1,
                              padding = "SAME", activation = tf.nn.relu, name = "p3.2" )

        in4   = tf.layers.max_pooling2d( X, pool_size = 2, strides = 1, padding = "SAME",
                                     name = "p4.1" )
        path4 = tf.layers.conv2d( in4, filters = np4_2, kernel_size = 1, strides = 1,
                              padding = "SAME", activation = tf.nn.relu, name = "p4.2" )

        return tf.concat( [path1, path2, path3, path4], axis = 3 )

def dogClassifier( X, y, inceptParams1, inceptParams2,
                   alpha = 0.001, b1 = 0.9, b2 = 0.999, epsilon = 1e-08 ):
    """Creates all objects needed for training the dog classifier."""

    with tf.name_scope( "encoder" ):

        conv1 = tf.layers.conv2d( X, filters = 16, kernel_size = 3, strides = 1,
                                  padding = "SAME", activation = tf.nn.relu,
                                  name = "conv1" )
        pool1 = tf.layers.max_pooling2d( conv1, pool_size = 2,
                                         strides = 2, name = "pool1" )

        lnr1 = tf.nn.local_response_normalization( pool1, depth_radius = 2, bias = 1,
                                                   alpha = 0.00002, beta = 0.75 )

        conv2 = tf.layers.conv2d( lnr1, filters = 32, kernel_size = 3, strides = 1,
                                  padding = "SAME", activation = tf.nn.relu,
                                  name = "conv2" )
        lnr2 = tf.nn.local_response_normalization( conv2, depth_radius = 2, bias = 1,
                                                   alpha = 0.00002, beta = 0.75 )
        pool2 = tf.layers.max_pooling2d( lnr2, pool_size = 2,
                                         strides = 2, name = "pool2" )

        incept1 = inceptionBlock( pool2, **inceptParams1 )
        pool3 = tf.layers.max_pooling2d( incept1, pool_size = 2,
                                         strides = 2, name = "pool3" )

        incept2 = inceptionBlock( pool3, **inceptParams2 )
        pool4 = tf.layers.max_pooling2d( incept2, pool_size = 2,
                                         strides = 2, name = "pool4" )

        flat = tf.layers.flatten( pool4 )

        logits = tf.layers.dense( flat, 120, name = "output" )

    with tf.name_scope("loss"):
        crossEnt = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = y, logits = logits)
        loss = tf.reduce_mean( crossEnt, name = "loss" )

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )

    with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer( learning_rate = alpha,
                                      beta1 = b1, beta2 = b2
                                      epsilon = epsilon )
        training = opt.minimize( loss )
        lossSummary = tf.summary.scalar("crossEntropy", loss)

    with tf.name_scope("utility"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return loss, training, accuracy, lossSummary, init, saver

def trainModel( trainX, trainY, valX, valY, params, saveModel = False ):

    valLoadedX, valLoadedY = genData( valX, valY, size = 400 )

    parameters = params[ "params" ]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (None, 400, 400, 1), name = "X")
    y = tf.placeholder(tf.int32, shape = (None), name = "y")

    loss, training, accuracy, lossSummary, init, saver = dogClassifier( X, y, **parameters )

    nEpochs = 5000
    batchSize = params[ "batchSize" ]

    loVal = 10000
    patience = 0

    step = 0

    with tf.Session() as sess:

        init.run()

        tls = [ loss.eval( feed_dict = { X : trainX, y : trainY } ) ]
        vls = [ loss.eval( feed_dict = { X : valLoadedX, y : valLoadedY } ) ]

        for epoch in range(nEpochs):
            for batchX, batchY in genBatch( trainX, trainY, batchSize, imgSize = 400 ):

                sess.run( training, feed_dict = { X : batchX, y : batchY } )
                step += 1

                if ( step % 50 == 0 ):
                    valLoss = loss.eval( feed_dict = { X : valLoadedX, y : valLoadedY } )

                    if ( valLoss < loVal ):
                        loVal = valLoss

                        if (saveModel):
                            saver.save( sess, "./best/mnist-best.ckpt" )

            trainLoss = loss.eval( feed_dict = { X : batchX, y : batchY } )
            valLoss   = loss.eval( feed_dict = { X : valLoadedX, y : valLoadedX } )

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
            print("Final model validation accuracy:", accuracy.eval(feed_dict = { X : valX, y : valY }) )
            print("***")
            print()

    return (loVal, tls, vls)

def crossValidation( trainX, trainY, k, params ):

    res = []
    kf = KFold( n_splits = k )

    for indTr, indVl in kf.split( trainX, trainY ):
        valLoss , tls, vls = trainModel( trainX[indTr], trainY[indTr],
                                         trainX[indVl], trainY[indVl], params )

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

    trX, valX, trY, valY = train_test_split( trainX, trainY, test_size = 5000,
                                                   random_state = 123 )

    loVal, trHist, valHist = trainModel( trX, trY, valX, valY, bestParams, saveModel = True )

    return ( loVal, trHist, valHist, bestParams )

