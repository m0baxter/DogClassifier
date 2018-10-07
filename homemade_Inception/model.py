
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from dogFunctions import genData, genBatch


def convBlock( X, trn, nFilters, kernelSize, bnm ):
    '''A block consisting of a convolution, a poolingi, and a batch normalization layer.'''

    heInit = tf.variance_scaling_initializer()

    with tf.name_scope("ConvolutionBlock"):

        conv = tf.layers.conv2d( X, filters = nFilters, kernel_size = kernelSize, strides = 1,
                                 padding = "same", kernel_initializer = heInit )
        pool = tf.layers.max_pooling2d( conv, pool_size = 2, strides = 2, padding = "valid",
                                        name = "pool" )
        bn = tf.layers.batch_normalization( pool, training = trn, momentum = bnm )

        return tf.nn.elu( bn )

def inceptionBlock( X, trn, nPath1, nPath2_1, nPath2_2, nPath3_1, nPath3_2, nPath4, bnm ):
    """Creates an inception block with a residual (skip) connection."""

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

        return tf.nn.elu( bn )

def dogClassifier( X, y, trn, alpha = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08,
                              bnm = 0.99, dropProb = 0.5 ):
    """Creates all objects needed for training the dog classifier."""

    heInit = tf.variance_scaling_initializer()

    with tf.name_scope( "cnn" ):

        conv1 = convBlock( X, trn, 8, 3, bnm )
        incept1 = inceptionBlock( conv1, trn, 4, 2, 4, 2, 4, 4, bnm )

        conv2     = convBlock( incept1, trn, 16, 3, bnm )
        incept2_1 = inceptionBlock( conv2,     trn, 8, 4, 8, 4, 8, 8, bnm )
        incept2_2 = inceptionBlock( incept2_1, trn, 8, 4, 8, 4, 8, 8, bnm )

        conv3     = convBlock( incept2_2, trn, 32, 3, bnm )
        incept3_1 = inceptionBlock( conv3,     trn, 16, 8, 16, 8, 16, 16, bnm )
        incept3_2 = inceptionBlock( incept3_1, trn, 16, 8, 16, 8, 16, 16, bnm )

        conv4     = convBlock( incept3_2, trn, 48, 3, bnm )
        incept4_1 = inceptionBlock( conv4,     trn, 24, 12, 24, 12, 24, 24, bnm )
        incept4_2 = inceptionBlock( incept4_1, trn, 24, 12, 24, 12, 24, 24, bnm )

        conv5     = convBlock( incept4_2, trn, 64, 3, bnm )
        incept5_1 = inceptionBlock( conv5,     trn, 32, 16, 32, 16, 32, 32, bnm )
        incept5_2 = inceptionBlock( incept5_1, trn, 32, 16, 32, 16, 32, 32, bnm )

        flat = tf.layers.flatten( incept5_2 )

        dropout1 = tf.layers.dropout( flat, dropProb, training = trn )

        fc1 = tf.layers.dense( dropout1, 1024, name = "fc1", kernel_initializer = heInit,
                               activation = tf.nn.elu )
        dropout2 = tf.layers.dropout( fc1, dropProb, training = trn )

        fc2 = tf.layers.dense( dropout2, 1024, name = "fc2", kernel_initializer = heInit,
                               activation = tf.nn.elu )
        dropout3 = tf.layers.dropout( fc2, dropProb, training = trn )

        logits = tf.layers.dense( dropout3, 120, name = "output", kernel_initializer = heInit )

        predict = tf.nn.softmax( logits )

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

    return loss, training, accuracy, predict, lossSummary, init, saver

def batchEval( X, y, allX, allY, batchSize, function ):

    count = 0
    temp  = 0

    for start in range(0, len(allX) - 1, batchSize):
        bX, bY = genData( allX[ start : start + batchSize ], allY, size = 200 )
        n = len(bX)
        count += n

        temp += function.eval( feed_dict = { X : bX, y : bY } ) * n

    return temp / count

def trainModel( trainX, valX, labels, params, saveModel = False , loadState = False,
                                                                  histDtype = np.float32 ):
    """Trains the model. Loads a previously saved state if loadStat is True."""

    imgSize = 200

    parameters = params[ "params" ]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (None, imgSize, imgSize, 3), name = "X")
    y = tf.placeholder(tf.int32, shape = (None), name = "y")
    trn = tf.placeholder_with_default( False, shape = (), name = "trn" )

    loss, training, accuracy, predict, lossSummary, init, saver = dogClassifier( X, y, trn, **parameters )

    extraOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    batchSize = params[ "batchSize" ]
    maxBatch = 512

    patience = 0
    step = 0

    startTime = time.time()

    with tf.Session() as sess:

        if ( loadState ):
            saver.restore( sess, "./best/dogClass-best.ckpt" )

        else:
            init.run()

        tls = [ histDtype( batchEval( X, y, trainX, labels, batchSize, loss     ) ) ]
        tas = [ histDtype( batchEval( X, y, trainX, labels, batchSize, accuracy ) ) ]
        vls = [ histDtype( batchEval( X, y, valX,   labels, batchSize, loss     ) ) ]
        vas = [ histDtype( batchEval( X, y, valX,   labels, batchSize, accuracy ) ) ]

        loVal = vls[0]

        while ( batchSize <= maxBatch ):
            for batchX, batchY in genBatch( trainX, labels, batchSize, imgSize = imgSize ):

                sess.run( [training, extraOps], feed_dict = { X : batchX, y : batchY, trn : True } )
                step += 1

                if ( step % 25 == 0 ):
                    valLoss   = batchEval( X, y, valX, labels, batchSize, loss     )
                    valAcc    = batchEval( X, y, valX, labels, batchSize, accuracy )
                    trainLoss = loss.eval(     feed_dict = { X : batchX, y : batchY } )
                    trAcc     = accuracy.eval( feed_dict = { X : batchX, y : batchY } )

                    tls.append( histDtype( trainLoss ) )
                    vls.append( histDtype( valLoss   ) )
                    tas.append( histDtype( trAcc     ) )
                    vas.append( histDtype( valAcc    ) )

                    print( ("Step {0}:\n   "\
                            "valLoss: {1:8.6f}, "\
                            "trainLoss: {2:8.6f}, "\
                            "trAcc: {3:5.4f}, "\
                            "valAcc: {4:5.4f}, "\
                            "patience: {5:>2d}").format( step, valLoss, trainLoss,
                                                         trAcc, valAcc, patience ) )

                    if ( valLoss < loVal ):
                        loVal = valLoss
                        patience = 0

                        if (saveModel):
                            saver.save( sess, "./best/dogClass-best.ckpt" )

                    elif ( step > 500 and valLoss >= loVal ):
                        patience += 1

                        if ( patience >= 10 ):
                            batchSize = 2 * batchSize
                            patience = 0
                            print( "\n\nbatchSize =", batchSize, "\n\n" )

    endTime = time.time()
    print( "Training time: {0:3.2f}h".format( (endTime - startTime) /3600.0 ) )

    return ( np.array(loVal), np.array(tls), np.array(vls), np.array(tas), np.array(vas) )

