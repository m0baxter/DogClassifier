
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from dogFunctions import genData, genBatch


def convBlock( X, trn, nFilters, kernelSize, bnm ):

    heInit = tf.variance_scaling_initializer()

    with tf.name_scope("ConvolutionBlock"):

        conv = tf.layers.conv2d( X, filters = nFilters, kernel_size = kernelSize, strides = 1,
                                 padding = "same", kernel_initializer = heInit )
        pool = tf.layers.max_pooling2d( conv, pool_size = 2, strides = 2, padding = "valid",
                                        name = "pool" )
        bn = tf.layers.batch_normalization( pool, training = trn, momentum = bnm )

        return tf.nn.elu( bn )

def inceptionBlock( X, trn, nPath1, nPath2_1, nPath2_2, nPath3_1, nPath3_2, nPath4, bnm ):
    """Creates an inception block with a residual connection."""

    heInit = tf.variance_scaling_initializer()

    m, h, w, c = X.get_shape().as_list()

    with tf.name_scope( "inceptionBlock" ):
        path1 = tf.layers.conv2d( X, filters = nPath1, kernel_size = 1, strides = 1,
                                  padding = "same", activation = tf.nn.relu,
                                  kernel_initializer = heInit )

        in2   = tf.layers.conv2d( X,   filters = nPath2_1, kernel_size = 1, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  kernel_initializer = heInit )
        path2 = tf.layers.conv2d( in2, filters = nPath2_2, kernel_size = 3, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  kernel_initializer = heInit )

        in3   = tf.layers.conv2d( X,   filters = nPath3_1, kernel_size = 1, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  kernel_initializer = heInit )
        path3 = tf.layers.conv2d( in3, filters = nPath3_2, kernel_size = 5, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  kernel_initializer = heInit )

        in4   = tf.layers.max_pooling2d( X, pool_size = 2, strides = 1, padding = "same",
                                     name = "p4.1" )
        path4 = tf.layers.conv2d( in4, filters = nPath4, kernel_size = 1, strides = 1,
                                  padding = "same", activation = tf.nn.elu,
                                  kernel_initializer = heInit)

        concat = tf.concat( [path1, path2, path3, path4], axis = 3 )

        merge = tf.layers.conv2d( concat, filters = c, kernel_size = 1, strides = 1,
                                  padding = "same", kernel_initializer = heInit )

        bn = tf.layers.batch_normalization( merge + X, training = trn, momentum = bnm )

        return tf.nn.elu( X + bn )

def dogClassifier( X, y, trn, alpha = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, bnm = 0.99 ):
    """Creates all objects needed for training the dog classifier."""

    heInit = tf.variance_scaling_initializer()

    with tf.name_scope( "cnn" ):

        conv1 = convBlock( X, trn, 8, 3, bnm )
        incept1 = inceptionBlock( conv1, trn, 4, 4, 4, 4, 4, 4, bnm )
        conv2 = convBlock( conv1, trn, 16, 3, bnm )
        incept2 = inceptionBlock( conv2, trn, 8, 4, 8, 4, 8, 8, bnm )
        conv3 = convBlock( incept2, trn, 32, 3, bnm )
        incept3 = inceptionBlock( conv3, trn, 16, 8, 16, 8, 16, 16, bnm )

        flat = tf.layers.flatten( incept3 )

        dropout1 = tf.layers.dropout( flat, 0.5, training = trn )

        fc1 = tf.layers.dense( dropout1, 1024, name = "fc1", kernel_initializer = heInit,
                               activation = tf.nn.elu )
        dropout2 = tf.layers.dropout( fc1, 0.5, training = trn )

        fc2 = tf.layers.dense( dropout2, 512, name = "fc2", kernel_initializer = heInit,
                               activation = tf.nn.elu )
        dropout3 = tf.layers.dropout( fc2, 0.5, training = trn )

        logits = tf.layers.dense( dropout3, 120, name = "output", kernel_initializer = heInit )

    with tf.name_scope("loss"):
        crossEnt = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = y, logits = logits)
        loss = tf.reduce_mean( crossEnt, name = "loss" )

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )

    with tf.name_scope("train"):
        #opt = tf.train.MomentumOptimizer( learning_rate = alpha, momentum = b1,
        #                                  use_nesterov = False )

        opt = tf.train.AdamOptimizer( learning_rate = alpha, beta1 = b1, beta2 = b2,
                                      epsilon = eps )


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

    extraOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    batchSize = params[ "batchSize" ]

    loVal = 10000
    patience = 0

    step = 0

    with tf.Session() as sess:

        init.run()

        tls = [ batchEval( X, y, trainX, labels, batchSize, loss ) ]
        tas = [ batchEval( X, y, trainX, labels, batchSize, accuracy ) ]
        vls = [ batchEval( X, y, valX,   labels, batchSize, loss ) ]
        vas = [ batchEval( X, y, valX,   labels, batchSize, accuracy ) ]

        while ( True ):
            for batchX, batchY in genBatch( trainX, labels, batchSize, imgSize = 200 ):

                sess.run( [training, extraOps], feed_dict = { X : batchX, y : batchY, trn : True } )
                step += 1

                if ( step % 25 == 0 ):
                    valLoss   = batchEval( X, y, valX, labels, batchSize, loss )
                    valAcc    = batchEval( X, y, valX, labels, batchSize, accuracy )
                    trainLoss = loss.eval( feed_dict = { X : batchX, y : batchY } )
                    trAcc     = batchEval( X, y, trainX, labels, batchSize, accuracy )

                    tls.append( trainLoss )
                    vls.append( valLoss )
                    tas.append( trAcc )
                    vas.append( valAcc )

                    if ( valLoss < loVal ):
                        loVal = valLoss
                        patience = 0

                        if (saveModel):
                            saver.save( sess, "./best/dogClass-best.ckpt" )

                    elif ( step > 500 and valLoss >= loVal):
                        patience += 1

                    print( ("Step {0}:\n   "\
                            "valLoss: {1:8.6f}, "\
                            "trainLoss: {2:8.6f}, "\
                            "trAcc: {3:5.4f}, "\
                            "valAcc: {4:5.4f}, "\
                            "patience: {5:>2d}").format( step, valLoss, trainLoss,
                                                         trAcc, valAcc, patience ) )

                    if ( patience >= 10 ): #20?

                        #print("\n***")
                        #print("Final validation accuracy:", batchEval( X, y, valX,
                        #                                               labels, batchSize,
                        #                                               accuracy ) )
                        #print("***\n")

                        return (loVal, tls, vls, tas, vas )

def crossValidation( trainX, trainY, k, params ):

    res = []
    kf = KFold( n_splits = k )

    for indTr, indVl in kf.split( trainX, trainY ):
        valLoss, tls, vls = trainModel( trainX[indTr], trainY,
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

