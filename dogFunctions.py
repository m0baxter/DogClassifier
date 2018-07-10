
import imageTrans as it
import numpy as np
import re

from keras.preprocessing import image


def readBreeds():
    """Reads an ordered list of dog breeds."""

    breeds = []

    with open( "breeds.txt", "r" ) as readFile:
        for line in readFile:
            breeds.append( line.strip() )

    return breeds

def getImageId( fileName ):
    """Collect the image Ids from the files."""

    return re.split('\.|/', fileName )[3]

def groupBreeds( files, labels ):

    breeds = [ [] for _ in range(120) ]

    for f in files:

        imgID = getImageId( f )
        breedId = labels[ labels["id"] == imgID ].breed.item()

        breeds[ breedId ].append( f )

    return breeds

def sampleBreeds( breed, dogs, N ):
    """Returns N samples from dogs that are breed."""

    return np.random.choice( dogs[breed], N, replace = False)

def sampleDogs( X, labels, valFrac ):
    """Selects N samples of each breed."""

    dogs = groupBreeds( X, labels )

    samples = np.array([], dtype = '<U44')

    for i in range(120):

        breedCount = (labels['breed'] == i).sum()

        samples = np.append( samples, sampleBreeds( i, dogs, int(valFrac * breedCount) ) )

    np.random.shuffle(samples)

    return samples

def genData( files, labels, size = 200 ):
    """Generates a set of training data and its associated labels."""

    X = np.zeros( (len(files), size, size, 3) )
    y = np.zeros( len(files), dtype = int )

    i = 0

    for f in files:
        imgID = getImageId( f )
        img = image.load_img( f, target_size = (size, size) )
        img = image.img_to_array(img)/255

        X[i] = img
        y[i] = labels[ labels["id"] == imgID ].breed.item()

        i += 1

    return X, y

def genBatch( files, labels, batchSize, imgSize = 200 ):
    """Generator of mini batches for training."""

    inds = np.random.permutation( len(files) )

    for start in range(0, len(files) - 1, batchSize):

        X, y = genData( files[ inds[start : start + batchSize] ], labels, size = imgSize )

        #Choose transformations
        #brightness:
        if ( np.random.rand() < 0.3 ):
            X = it.adjustBrightness( X, np.random.uniform(-0.15, 0.15) )

        #other transforms:
        #if ( np.random.rand() < 0.5 ):
        #    imageTransformers = [ lambda x : it.scaleImages( x, np.random.uniform(0.86, 1.18) ),
        #                          lambda x : it.translateImages( x, np.random.randint(0,5),
        #                                                            np.random.uniform(0, 0.15) ),
        #                          #lambda x : it.rotateK90Degs( x, np.random.randint(0,4) ),
        #                          lambda x : it.rotateImages( x, np.random.uniform(-np.pi/6, np.pi/6) ) ]

        #    transform = np.random.choice( imageTransformers )
        #    X = transform(X)

        #mirror flip:
        if ( np.random.rand() < 0.5 ):
            X = it.mirrorImages( X, 0 )

        yield X, y

