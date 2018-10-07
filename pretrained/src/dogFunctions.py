
import src.imageTrans as it
import numpy as np
import re

from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


def readSavedFiles( path ):

    files = []

    with open( path, "r" ) as readFile:
        for line in readFile:
            files.append( line.strip() )

    return np.array( files )

def writeFilesList( path, files ):
    
    with open( path, "w") as outFile:
        for f in files:
            outFile.write( f + "\n" )

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

    return X, to_categorical(y, num_classes = 120 )

def genBatch( files, labels, batchSize, imgSize = 200, train = False ):
    """Generator of mini batches for training."""
    
    while (True):
    
        inds = np.random.permutation( len(files) )
        
        for start in range(0, len(files) - 1, batchSize):
        
            X, y = genData( files[ inds[start : start + batchSize] ], labels, size = imgSize )
        
            if ( train ):
            
                #Choose transformations
                #brightness:
                if ( np.random.rand() < 0.5 ):
                    X = it.adjustBrightness( X, np.random.uniform(-0.15, 0.15) )

                #Scale:
                if ( np.random.rand() < 0.5 ):
                    X = it.scaleImages( X, np.random.uniform(0.86, 1.18) )
                    
                #Translate:
                if ( np.random.rand() < 0.5 ):
                    X = it.translateImages( X, np.random.randint(0,5), np.random.uniform(0, 0.15) )

                #Rotate:
                if ( np.random.rand() < 0.5 ):
                    X = it.rotateImages( X, np.random.uniform(-np.pi, np.pi) )

                #mirror flip:
                if ( np.random.rand() < 0.5 ):
                    X = it.mirrorImages( X, 0 )

            yield X, y
